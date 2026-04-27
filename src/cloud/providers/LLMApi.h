#ifndef LLMAPI_H
#define LLMAPI_H

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace LLMApi
{
    using json = nlohmann::json;

    // 视觉请求运行时配置。
    // 流水线模式只依赖这组字段，不需要暴露完整示例程序的全部配置。
    struct VisionConfig
    {
        std::string api_key;
        std::string base_url;
        std::string model = "gpt-5.4";
        std::string prompt = "请描述这张图片里的内容。";
        std::string image_detail = "auto";
    };

    // 这个类封装了一个最小可运行的 OpenAI Responses API 调用示例：
    // - 从配置文件读取参数
    // - 根据 text / vision 模式构造请求
    // - 通过 libcurl 发送 HTTP 请求
    // - 解析并输出模型返回结果
    class LLMApi
    {
    public:
        LLMApi();
        LLMApi(std::string api_key, std::string base_url);
        ~LLMApi();

        // 从配置文件中提取流水线视觉模式需要的字段。
        static bool LoadVisionConfig(const std::string &config_path,
                                     VisionConfig *config,
                                     std::string *error_message);

        // 更新当前实例使用的认证信息和服务地址。
        void Configure(std::string api_key, std::string base_url);

        // 发送单次视觉请求并返回提取后的文本结果。
        std::string AnalyzeVision(const std::string &model,
                                  const std::string &prompt,
                                  const std::string &image_path,
                                  const std::string &detail) const;

        // 程序主入口：
        // 解析命令行参数，读取配置，发起请求，并把结果输出到终端。
        int run(int argc, char *argv[]);

    private:
        // 当前示例支持两种请求模式：
        // - Text: 纯文本对话
        // - Vision: 文本 + 图片输入
        enum class RequestMode
        {
            Text,
            Vision,
        };

        // AppConfig 保存 config.json 中可配置的全部字段。
        // 这里的默认值让程序在字段缺失时仍有合理兜底。
        struct AppConfig
        {
            // OpenAI 或兼容服务的 API Key。
            std::string api_key;
            // API 基础地址，例如 https://api.openai.com/v1。
            std::string base_url;

            // 默认使用视觉模式，便于展示图片输入示例。
            RequestMode mode = RequestMode::Vision;

            // Text 模式配置。
            std::string text_model = "gpt-5.4";
            std::string text_prompt = "你好";

            // Vision 模式配置。
            std::string vision_model = "gpt-5.4";
            std::string vision_prompt = "请描述这张图片里的内容。";
            std::string image_path = "test.png";
            std::string image_detail = "auto";
        };

        // 判断某个配置项是不是“未填写占位值”。
        static bool is_placeholder(const std::string &value);
        // 从标准错误响应里提取更适合展示给用户的错误文本。
        static std::string format_api_error(const json &response_json);
        // 以二进制方式读取图片文件。
        static std::vector<unsigned char> read_binary_file(const std::string &path);
        // 判断文件是否存在。
        static bool file_exists(const std::string &path);
        // 判断路径是否为绝对路径。
        static bool is_absolute_path(const std::string &path);
        // 返回路径所在目录。
        static std::string dirname(const std::string &path);
        // 组合目录和文件路径。
        static std::string join_paths(const std::string &dir, const std::string &name);
        // 仅处理 ASCII 字母的小写转换，用于模式和扩展名归一化。
        static std::string to_lower_ascii(std::string value);
        // 把字符串模式名转换成内部枚举。
        static RequestMode parse_request_mode(std::string value);
        // 读取一个字符串类型配置字段；不存在时返回默认值。
        static std::string read_string_field(const json &root,
                                             const char *field_name,
                                             const std::string &default_value);
        // 把路径格式化成更适合输出给用户查看的样子。
        static std::string format_path_for_display(const std::string &path);
        // 根据命令行参数决定配置文件路径。
        static std::string resolve_config_path(int argc, char *argv[]);
        // 从 JSON 配置文件加载程序运行参数。
        static AppConfig load_app_config(const std::string &config_path);
        // 根据文件扩展名推断图片 MIME 类型。
        static std::string guess_mime_type(const std::string &path);
        // 把原始字节编码成 base64，供 data URL 使用。
        static std::string base64_encode(const std::vector<unsigned char> &bytes);
        // 把图片文件转换成 data:<mime>;base64,... 形式。
        static std::string build_image_data_url(const std::string &image_path);
        // 构造纯文本请求体。
        static json build_text_response_payload(const std::string &model,
                                                const std::string &input);
        // 构造图文请求体。
        static json build_vision_response_payload(const std::string &model,
                                                  const std::string &prompt,
                                                  const std::string &image_data_url,
                                                  const std::string &detail);
        // 把 HTTP 响应体解析为 JSON。
        static json parse_response_json(const std::string &response_body);
        // 从 Responses API 的返回结构中提取最终文本。
        static std::string extract_output_text(const json &response);
        // libcurl 写回调：把收到的数据追加到字符串缓冲区。
        static size_t write_callback(char *ptr, size_t size, size_t nmemb, void *userdata);

        // 统一创建 Responses API 请求。
        json create_response(const json &payload) const;
        // 创建纯文本请求。
        json create_text_response(const std::string &model,
                                  const std::string &input) const;
        // 创建视觉请求。
        json create_vision_response(const std::string &model,
                                    const std::string &prompt,
                                    const std::string &image_path,
                                    const std::string &detail) const;
        // 向指定 API 路径发送 JSON POST 请求，并返回原始响应体。
        std::string post_json(const std::string &path,
                              const std::string &payload) const;

        // 当前实例实际使用的认证信息和基础地址。
        std::string api_key_;
        std::string base_url_;
    };
}

#endif
