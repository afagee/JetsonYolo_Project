#ifndef LOGGING_H
#define LOGGING_H

#include <NvInfer.h>
#include <iostream>

// Logger class cho TensorRT (bắt buộc)
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

#endif // LOGGING_H

