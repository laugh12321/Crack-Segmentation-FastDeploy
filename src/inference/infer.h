#ifndef _INFER_H_
#define _INFER_H_

#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/vision.h"
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <vector>
#include <string>

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

class CrackInfer {
public:
    CrackInfer(const std::string& config_file);
    std::vector<fastdeploy::vision::SegmentationResult> batchinfer(const std::vector<std::string> &images_path, const std::string& save_dir);
private:
    fastdeploy::vision::segmentation::PaddleSegModel* model;
    fastdeploy::RuntimeOption option;
    void setRuntimeOption(const YAML::Node& config);
};

#endif // _INFER_H_