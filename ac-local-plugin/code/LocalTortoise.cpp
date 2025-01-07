// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/tortoise/Instance.hpp>
#include <ac/tortoise/Init.hpp>
#include <ac/tortoise/Model.hpp>

#include <ac/local/Instance.hpp>
#include <ac/local/Model.hpp>
#include <ac/local/ModelLoader.hpp>

#include <ac/schema/TortoiseCpp.hpp>
#include <ac/schema/DispatchHelpers.hpp>

#include <astl/move.hpp>
#include <astl/move_capture.hpp>
#include <astl/iile.h>
#include <astl/throw_stdex.hpp>
#include <astl/workarounds.h>

#include "aclp-tortoise-version.h"
#include "aclp-tortoise-interface.hpp"

namespace ac::local {

namespace {

class TortoiseInstance final : public Instance {
    std::shared_ptr<tortoise::Model> m_model;
    tortoise::Instance m_instance;
    schema::OpDispatcherData m_dispatcherData;
public:
    using Schema = ac::local::schema::TortoiseCppLoader::InstanceGeneral;
    using Interface = ac::local::schema::TortoiseCppInterface;

    TortoiseInstance(std::shared_ptr<tortoise::Model> model, tortoise::Instance::InitParams params)
        : m_model(astl::move(model))
        , m_instance(*m_model, astl::move(params))
    {
        schema::registerHandlers<Interface::Ops>(m_dispatcherData, *this);
    }

    Interface::OpTTS::Return on(Interface::OpTTS, Interface::OpTTS::Params params) {
        const auto& text = params.text.value();
        const auto& voicePath = params.voicePath.value();

        auto result = m_instance.textToSpeech(text, voicePath);

        ac::Blob resultBlob;
        resultBlob.resize(result.size() * sizeof(float));
        memcpy(resultBlob.data(), result.data(), resultBlob.size());

        return {
            .result = std::move(resultBlob)
        };
    }

    virtual Dict runOp(std::string_view op, Dict params, ProgressCb) override {
        auto ret = m_dispatcherData.dispatch(op, astl::move(params));
        if (!ret) {
            throw_ex{} << "tortoise: unknown op: " << op;
        }
        return *ret;
    }
};

class TortoiseModel final : public Model {
    std::shared_ptr<tortoise::Model> m_model;
public:
    using Schema = ac::local::schema::TortoiseCppLoader;

    TortoiseModel(
        std::string_view autoregressiveModelPath,
        std::string_view diffusionModelPath,
        std::string_view vocoderModelPath,
        tortoise::Model::Params params)
        : m_model(std::make_shared<tortoise::Model>(autoregressiveModelPath, diffusionModelPath, vocoderModelPath, astl::move(params)))
    {}

    virtual std::unique_ptr<Instance> createInstance(std::string_view, Dict params) override {
        tortoise::Instance::InitParams initParams;
        initParams.tokenizerPath = Dict_optValueAt(params, "tokenizerPath", std::string());
        initParams.seed = Dict_optValueAt(params, "seed", 42);
        return std::make_unique<TortoiseInstance>(m_model, astl::move(initParams));
    }
};

class TortoiseModelLoader final : public ModelLoader {
public:
    virtual const Info& info() const noexcept override {
        static Info i = {
            .name = "ac tortoise.cpp",
            .vendor = "Alpaca Core"
        };
        return i;
    }

    /// Check if the model can be loaded
    virtual bool canLoadModel(const ModelAssetDesc& desc, const Dict& /*params*/) const noexcept override {
         return desc.type == "tortoise.cpp";
    }

    virtual ModelPtr loadModel(ModelAssetDesc desc, Dict /*params*/, ProgressCb /*progressCb*/) override {
        if (desc.assets.size() != 3) throw_ex{} << "tortoise: expected exactly one local asset";
        auto& aggresive = desc.assets[0].path;
        auto& diffusion = desc.assets[1].path;
        auto& vocoder = desc.assets[2].path;
        tortoise::Model::Params modelParams;
        return std::make_shared<TortoiseModel>(aggresive, diffusion, vocoder, modelParams);
    }
};
}

} // namespace ac::local

namespace ac::tortoise {

void init() {
    initLibrary();
}

std::vector<ac::local::ModelLoaderPtr> getLoaders() {
    std::vector<ac::local::ModelLoaderPtr> ret;
    ret.push_back(std::make_unique<local::TortoiseModelLoader>());
    return ret;
}

local::PluginInterface getPluginInterface() {
    return {
        .label = "ac tortoise.cpp",
        .desc = "tortoise.cpp plugin for ac-local",
        .vendor = "Alpaca Core",
        .version = astl::version{
            ACLP_tortoise_VERSION_MAJOR, ACLP_tortoise_VERSION_MINOR, ACLP_tortoise_VERSION_PATCH
        },
        .init = init,
        .getLoaders = getLoaders,
    };
}

} // namespace ac::tortoise

