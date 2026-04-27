#include "LLMApi.h"

#include <cctype>
#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace
{
    // RAII 守卫：确保 libcurl 全局状态在程序生命周期内正确初始化和清理。
    class CurlGlobalGuard
    {
    public:
        CurlGlobalGuard()
        {
            const auto code = curl_global_init(CURL_GLOBAL_DEFAULT);
            if (code != CURLE_OK)
            {
                throw std::runtime_error(curl_easy_strerror(code));
            }
        }

        ~CurlGlobalGuard()
        {
            curl_global_cleanup();
        }
    };

    CurlGlobalGuard &GetCurlGlobalGuard()
    {
        static CurlGlobalGuard guard;
        return guard;
    }
}

namespace LLMApi
{
    LLMApi::LLMApi() = default;

    LLMApi::LLMApi(std::string api_key, std::string base_url)
        : api_key_(std::move(api_key)), base_url_(std::move(base_url))
    {
    }

    LLMApi::~LLMApi() = default;

    bool LLMApi::LoadVisionConfig(const std::string &config_path,
                                  VisionConfig *config,
                                  std::string *error_message)
    {
        if (config == NULL)
        {
            if (error_message != NULL)
            {
                *error_message = "VisionConfig output pointer is null";
            }
            return false;
        }

        const char *const fallback_prefix = "../";
        std::string resolved_config_path = config_path;

        try
        {
            AppConfig app_config;
            try {
                app_config = load_app_config(resolved_config_path);
            } catch (const std::exception &) {
                if (!config_path.empty() && !is_absolute_path(config_path)) {
                    resolved_config_path = std::string(fallback_prefix) + config_path;
                    app_config = load_app_config(resolved_config_path);
                } else {
                    throw;
                }
            }
            config->api_key = app_config.api_key;
            config->base_url = app_config.base_url;
            config->model = app_config.vision_model;
            config->prompt = app_config.vision_prompt;
            config->image_detail = app_config.image_detail;
            if (is_placeholder(config->api_key) || is_placeholder(config->base_url)) {
                throw std::runtime_error(
                    "LLM config is missing api_key or base_url: " + resolved_config_path);
            }
            return true;
        }
        catch (const std::exception &ex)
        {
            if (error_message != NULL)
            {
                *error_message = ex.what();
            }
            return false;
        }
    }

    void LLMApi::Configure(std::string api_key, std::string base_url)
    {
        api_key_ = std::move(api_key);
        base_url_ = std::move(base_url);
    }

    std::string LLMApi::AnalyzeVision(const std::string &model,
                                      const std::string &prompt,
                                      const std::string &image_path,
                                      const std::string &detail) const
    {
        const json response_json = create_vision_response(model, prompt, image_path, detail);
        const std::string text = extract_output_text(response_json);
        if (!text.empty())
        {
            return text;
        }
        return response_json.dump(2);
    }

    int LLMApi::run(int argc, char *argv[])
    {
        try
        {
            (void)GetCurlGlobalGuard();

            // 先确定配置文件位置，再把所有运行参数读进来。
            const std::string config_path = resolve_config_path(argc, argv);
            const AppConfig config = load_app_config(config_path);

            // 关键配置缺失时，尽早给出明确提示，避免后续错误更难排查。
            if (is_placeholder(config.api_key))
            {
                throw std::runtime_error(
                    "Please set \"api_key\" in config file before running: " +
                    format_path_for_display(config_path));
            }

            if (is_placeholder(config.base_url))
            {
                throw std::runtime_error(
                    "Please set \"base_url\" in config file before running: " +
                    format_path_for_display(config_path));
            }

            Configure(config.api_key, config.base_url);

            json response_json;
            if (config.mode == RequestMode::Vision)
            {
                // 视觉模式要求必须有图片文件路径。
                if (is_placeholder(config.image_path))
                {
                    throw std::runtime_error(
                        "Please set \"image_path\" in config file before using Vision mode: " +
                        format_path_for_display(config_path));
                }

                // 发送图文请求。
                response_json = create_vision_response(
                    config.vision_model,
                    config.vision_prompt,
                    config.image_path,
                    config.image_detail);
            }
            else
            {
                // 发送纯文本请求。
                response_json = create_text_response(
                    config.text_model,
                    config.text_prompt);
            }

            // 优先输出提取后的纯文本；如果结构不符合预期，再回退到完整 JSON。
            const std::string text = extract_output_text(response_json);
            if (!text.empty())
            {
                std::cout << text << '\n';
            }
            else
            {
                std::cout << response_json.dump(2) << '\n';
            }

            return 0;
        }
        catch (const std::exception &ex)
        {
            // 统一错误出口，方便在命令行直接看到修复方向。
            std::cerr << "Error: " << ex.what() << '\n'
                      << "Edit config.json (or pass a custom config path) and fill in:\n"
                      << "  - api_key\n"
                      << "  - base_url\n"
                      << "  - mode\n"
                      << "  - text_* or vision_* settings\n";
            return 1;
        }
    }

    bool LLMApi::is_placeholder(const std::string &value)
    {
        // 支持两种“未配置”判定：
        // - 空字符串
        // - 仍保留 REPLACE_WITH_* 之类的占位文本
        return value.empty() || value.find("REPLACE_WITH_") != std::string::npos;
    }

    std::string LLMApi::format_api_error(const json &response_json)
    {
        // 优先读取标准 error.message 字段。
        if (response_json.contains("error") &&
            response_json["error"].is_object() &&
            response_json["error"].contains("message") &&
            response_json["error"]["message"].is_string())
        {
            return response_json["error"]["message"].get<std::string>();
        }

        // 某些兼容服务可能直接把 error 作为字符串返回。
        if (response_json.contains("error") && response_json["error"].is_string())
        {
            return response_json["error"].get<std::string>();
        }

        // 实在没有统一格式时，直接把原始 JSON 打出来。
        return response_json.dump(2);
    }

    std::vector<unsigned char> LLMApi::read_binary_file(const std::string &path)
    {
        // 视觉模型需要发送图片内容，因此这里按原始字节读取。
        std::ifstream input(path.c_str(), std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("Failed to open image file: " + path);
        }

        return std::vector<unsigned char>(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    }

    bool LLMApi::file_exists(const std::string &path)
    {
        std::ifstream input(path.c_str(), std::ios::binary);
        return input.good();
    }

    bool LLMApi::is_absolute_path(const std::string &path)
    {
        return !path.empty() && path[0] == '/';
    }

    std::string LLMApi::dirname(const std::string &path)
    {
        const std::string::size_type pos = path.find_last_of('/');
        if (pos == std::string::npos)
        {
            return ".";
        }
        if (pos == 0)
        {
            return "/";
        }
        return path.substr(0, pos);
    }

    std::string LLMApi::join_paths(const std::string &dir, const std::string &name)
    {
        if (dir.empty() || dir == ".")
        {
            return name;
        }
        if (dir[dir.size() - 1] == '/')
        {
            return dir + name;
        }
        return dir + "/" + name;
    }

    std::string LLMApi::to_lower_ascii(std::string value)
    {
        // 这里只处理 ASCII 范围内的大小写转换，足够覆盖 mode / 扩展名等字段。
        for (std::string::size_type i = 0; i < value.size(); ++i)
        {
            value[i] = static_cast<char>(
                std::tolower(static_cast<unsigned char>(value[i])));
        }
        return value;
    }

    LLMApi::RequestMode LLMApi::parse_request_mode(std::string value)
    {
        // 允许用户在配置里写成 Text / TEXT / vision 等不同大小写形式。
        value = to_lower_ascii(std::move(value));
        if (value == "text")
        {
            return RequestMode::Text;
        }
        if (value == "vision")
        {
            return RequestMode::Vision;
        }

        throw std::runtime_error(
            "Invalid config field \"mode\": expected \"text\" or \"vision\"");
    }

    std::string LLMApi::read_string_field(const json &root,
                                          const char *field_name,
                                          const std::string &default_value)
    {
        // 字段缺失时走默认值，便于让配置文件保持最小化。
        const json::const_iterator it = root.find(field_name);
        if (it == root.end())
        {
            return default_value;
        }
        // 明确要求字段必须是字符串，避免类型错误静默传播。
        if (!it->is_string())
        {
            throw std::runtime_error(
                "Config field \"" + std::string(field_name) + "\" must be a string");
        }
        return it->get<std::string>();
    }

    std::string LLMApi::format_path_for_display(const std::string &path)
    {
        return path;
    }

    std::string LLMApi::resolve_config_path(int argc, char *argv[])
    {
        // 程序只接受 0 或 1 个附加参数：自定义配置文件路径。
        if (argc > 2)
        {
            const std::string exe_name =
                (argc > 0 && argv[0] != NULL && argv[0][0] != '\0') ? argv[0] : "openai_api_example";
            throw std::runtime_error("Usage: " + exe_name + " [config.json]");
        }

        if (argc == 2 && argv[1] != NULL && argv[1][0] != '\0')
        {
            return argv[1];
        }

        // 不显式传参时，先找当前目录，再找上一级目录。
        const char *const candidates[] = {
            "config.json",
            "../config.json",
        };
        for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i)
        {
            if (file_exists(candidates[i]))
            {
                return candidates[i];
            }
        }

        return "config.json";
    }

    LLMApi::AppConfig LLMApi::load_app_config(const std::string &config_path)
    {
        // 读取并解析 JSON 配置文件。
        std::ifstream input(config_path.c_str());
        if (!input)
        {
            throw std::runtime_error(
                "Failed to open config file: " + format_path_for_display(config_path));
        }

        json root;
        try
        {
            input >> root;
        }
        catch (const json::parse_error &ex)
        {
            throw std::runtime_error(
                "Failed to parse config file \"" + format_path_for_display(config_path) +
                "\" as JSON: " + ex.what());
        }

        if (!root.is_object())
        {
            throw std::runtime_error(
                "Config file must contain a JSON object: " +
                format_path_for_display(config_path));
        }

        AppConfig config;
        // 逐项读取可配置字段；不存在时继续使用结构体中的默认值。
        config.api_key = read_string_field(root, "api_key", config.api_key);
        config.base_url = read_string_field(root, "base_url", config.base_url);
        config.text_model = read_string_field(root, "text_model", config.text_model);
        config.text_prompt = read_string_field(root, "text_prompt", config.text_prompt);
        config.vision_model = read_string_field(root, "vision_model", config.vision_model);
        config.vision_prompt = read_string_field(root, "vision_prompt", config.vision_prompt);
        config.image_path = read_string_field(root, "image_path", config.image_path);
        config.image_detail = read_string_field(root, "image_detail", config.image_detail);

        const json::const_iterator mode_it = root.find("mode");
        if (mode_it != root.end())
        {
            if (!mode_it->is_string())
            {
                throw std::runtime_error(
                    "Config field \"mode\" must be a string: \"text\" or \"vision\"");
            }
            config.mode = parse_request_mode(mode_it->get<std::string>());
        }

        // 相对图片路径按配置文件所在目录解析，避免必须从固定工作目录启动程序。
        if (!config.image_path.empty() && !is_absolute_path(config.image_path))
        {
            config.image_path = join_paths(dirname(config_path), config.image_path);
        }

        return config;
    }

    std::string LLMApi::guess_mime_type(const std::string &path)
    {
        // data URL 需要显式 MIME 类型，因此根据扩展名做简单推断。
        const std::string::size_type dot = path.find_last_of('.');
        if (dot == std::string::npos)
        {
            throw std::runtime_error("Image path has no file extension: " + path);
        }

        const std::string ext = to_lower_ascii(path.substr(dot + 1));
        if (ext == "png")
        {
            return "image/png";
        }
        if (ext == "jpg" || ext == "jpeg")
        {
            return "image/jpeg";
        }
        if (ext == "webp")
        {
            return "image/webp";
        }
        if (ext == "gif")
        {
            return "image/gif";
        }

        // 目前只支持示例里最常见的几种图片格式。
        throw std::runtime_error(
            "Unsupported image extension: ." + ext +
            " (supported: png, jpg, jpeg, webp, gif)");
    }

    std::string LLMApi::base64_encode(const std::vector<unsigned char> &bytes)
    {
        // 手写一个最小可用的 base64 编码器，避免额外依赖。
        static const char kAlphabet[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string encoded;
        encoded.reserve(((bytes.size() + 2) / 3) * 4);

        // 每 3 个字节编码成 4 个 base64 字符；不足部分用 '=' 补齐。
        for (std::size_t i = 0; i < bytes.size(); i += 3)
        {
            const unsigned int octet_a = bytes[i];
            const unsigned int octet_b = (i + 1 < bytes.size()) ? bytes[i + 1] : 0U;
            const unsigned int octet_c = (i + 2 < bytes.size()) ? bytes[i + 2] : 0U;
            const unsigned int triple = (octet_a << 16) | (octet_b << 8) | octet_c;

            encoded.push_back(kAlphabet[(triple >> 18) & 0x3F]);
            encoded.push_back(kAlphabet[(triple >> 12) & 0x3F]);
            encoded.push_back((i + 1 < bytes.size()) ? kAlphabet[(triple >> 6) & 0x3F] : '=');
            encoded.push_back((i + 2 < bytes.size()) ? kAlphabet[triple & 0x3F] : '=');
        }

        return encoded;
    }

    std::string LLMApi::build_image_data_url(const std::string &image_path)
    {
        // Responses API 支持直接传 data URL，这样无需单独上传图片文件。
        const std::vector<unsigned char> bytes = read_binary_file(image_path);
        const std::string mime_type = guess_mime_type(image_path);
        return "data:" + mime_type + ";base64," + base64_encode(bytes);
    }

    json LLMApi::build_text_response_payload(const std::string &model,
                                             const std::string &input)
    {
        // Text 模式下，最简单的请求体只需要模型名和输入文本。
        return json{
            {"model", model},
            {"input", input},
        };
    }

    json LLMApi::build_vision_response_payload(const std::string &model,
                                               const std::string &prompt,
                                               const std::string &image_data_url,
                                               const std::string &detail)
    {
        // 视觉输入在 Responses API 中表现为 content 数组中的 input_image 项。
        json image_item = {
            {"type", "input_image"},
            {"image_url", image_data_url},
        };

        // detail 是可选参数，用来提示模型如何处理图像细节。
        if (!detail.empty())
        {
            image_item["detail"] = detail;
        }

        // 图文请求把文本和图片放在同一条 user 消息里。
        return json{
            {"model", model},
            {"input", json::array({
                {
                    {"role", "user"},
                    {"content", json::array({
                        {
                            {"type", "input_text"},
                            {"text", prompt},
                        },
                        image_item,
                    })},
                },
            })},
        };
    }

    json LLMApi::parse_response_json(const std::string &response_body)
    {
        try
        {
            return json::parse(response_body);
        }
        catch (const json::parse_error &ex)
        {
            // 一旦后端返回 HTML、纯文本或损坏 JSON，这里的错误信息能帮助快速排查。
            throw std::runtime_error(
                std::string("Failed to parse response as JSON: ") + ex.what() +
                "\nRaw response:\n" + response_body);
        }
    }

    std::string LLMApi::extract_output_text(const json &response)
    {
        // 某些响应会直接给出 output_text，优先走这个快捷字段。
        if (response.contains("output_text") && response["output_text"].is_string())
        {
            return response["output_text"].get<std::string>();
        }

        // 否则退回到完整 output 数组里逐项提取 output_text 内容。
        if (!response.contains("output") || !response["output"].is_array())
        {
            return std::string();
        }

        std::ostringstream text;
        for (json::const_iterator output_it = response["output"].begin();
             output_it != response["output"].end();
             ++output_it)
        {
            const json &output_item = *output_it;
            // Responses API 的每个 output 项下可能还包含多个 content 段。
            if (!output_item.is_object())
            {
                continue;
            }
            if (!output_item.contains("content") || !output_item["content"].is_array())
            {
                continue;
            }

            for (json::const_iterator content_it = output_item["content"].begin();
                 content_it != output_item["content"].end();
                 ++content_it)
            {
                const json &content_item = *content_it;
                if (!content_item.is_object())
                {
                    continue;
                }
                if (content_item.value("type", "") != "output_text")
                {
                    continue;
                }
                if (!content_item.contains("text") || !content_item["text"].is_string())
                {
                    continue;
                }

                // 多段文本之间用换行拼接，保留可读性。
                if (text.tellp() > 0)
                {
                    text << '\n';
                }
                text << content_item["text"].get<std::string>();
            }
        }

        return text.str();
    }

    size_t LLMApi::write_callback(char *ptr, size_t size, size_t nmemb, void *userdata)
    {
        // libcurl 每收到一段响应数据都会回调这里。
        std::string *buffer = static_cast<std::string *>(userdata);
        buffer->append(ptr, size * nmemb);
        return size * nmemb;
    }

    json LLMApi::create_response(const json &payload) const
    {
        // 当前示例统一调用 /responses 接口。
        const std::string response_body = post_json("/responses", payload.dump());
        return parse_response_json(response_body);
    }

    json LLMApi::create_text_response(const std::string &model,
                                      const std::string &input) const
    {
        // 文本请求只是构造 payload 的方式不同，真正发送仍走统一接口。
        return create_response(build_text_response_payload(model, input));
    }

    json LLMApi::create_vision_response(const std::string &model,
                                        const std::string &prompt,
                                        const std::string &image_path,
                                        const std::string &detail) const
    {
        // 视觉请求先把图片文件编码成 data URL，再拼进 JSON 请求体。
        const std::string image_data_url = build_image_data_url(image_path);
        return create_response(
            build_vision_response_payload(model, prompt, image_data_url, detail));
    }

    std::string LLMApi::post_json(const std::string &path,
                                  const std::string &payload) const
    {
        (void)GetCurlGlobalGuard();

        CURL *curl = curl_easy_init();
        if (curl == NULL)
        {
            throw std::runtime_error("curl_easy_init failed");
        }

        // 这里准备完整请求 URL、认证头和响应缓冲区。
        struct curl_slist *headers = NULL;
        std::string response_body;
        const std::string url = base_url_ + path;
        const std::string authorization = "Authorization: Bearer " + api_key_;

        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, authorization.c_str());

        // 配置一个标准 JSON POST 请求。
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl,
                         CURLOPT_POSTFIELDSIZE_LARGE,
                         static_cast<curl_off_t>(payload.size()));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "openai_api_example/0.1");

        // 真正执行网络请求。
        const CURLcode result = curl_easy_perform(curl);
        long status_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);

        // 无论成功失败，都先释放 libcurl 资源。
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (result != CURLE_OK)
        {
            // 例如连接失败、超时、TLS 握手异常，都会从这里抛出。
            throw std::runtime_error(
                std::string("curl request failed: ") + curl_easy_strerror(result));
        }

        if (status_code >= 400)
        {
            try
            {
                // 如果服务端返回了标准 JSON 错误结构，就尽量格式化后再抛出。
                const json response_json = parse_response_json(response_body);
                throw std::runtime_error(
                    "OpenAI API returned HTTP " + std::to_string(status_code) +
                    "\n" + format_api_error(response_json));
            }
            catch (const std::runtime_error &)
            {
                throw;
            }
            catch (...)
            {
                // 对于非 JSON 错误页，直接附上原始响应文本。
                throw std::runtime_error(
                    "OpenAI API returned HTTP " + std::to_string(status_code) +
                    "\n" + response_body);
            }
        }

        return response_body;
    }
}
