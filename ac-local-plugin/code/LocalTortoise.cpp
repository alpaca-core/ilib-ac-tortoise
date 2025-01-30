// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/tortoise/Instance.hpp>
#include <ac/tortoise/Init.hpp>
#include <ac/tortoise/Model.hpp>

#include <ac/local/Provider.hpp>

#include <ac/schema/TortoiseCpp.hpp>
#include <ac/schema/OpDispatchHelpers.hpp>

#include <ac/frameio/SessionCoro.hpp>
#include <ac/FrameUtil.hpp>

#include <astl/move.hpp>
#include <astl/move_capture.hpp>
#include <astl/iile.h>
#include <astl/throw_stdex.hpp>
#include <astl/workarounds.h>

#include "aclp-tortoise-version.h"
#include "aclp-tortoise-interface.hpp"

namespace ac::local {

namespace {

// class TortoiseInstance final : public Instance {
//     std::shared_ptr<tortoise::Model> m_model;
//     tortoise::Instance m_instance;
//     schema::OpDispatcherData m_dispatcherData;
// public:
//     using Schema = ac::schema::TortoiseProvider::InstanceGeneral;
//     using Interface = ac::schema::TortoiseCppInterface;

//     TortoiseInstance(std::shared_ptr<tortoise::Model> model, tortoise::Instance::InitParams params)
//         : m_model(astl::move(model))
//         , m_instance(*m_model, astl::move(params))
//     {
//         schema::registerHandlers<Interface::Ops>(m_dispatcherData, *this);
//     }

//     Interface::OpTTS::Return on(Interface::OpTTS, Interface::OpTTS::Params params) {
//         const auto& text = params.text.value();
//         const auto& voicePath = params.voicePath.value();

//         auto result = m_instance.textToSpeech(text, voicePath);

//         ac::Blob resultBlob;
//         resultBlob.resize(result.size() * sizeof(float));
//         memcpy(resultBlob.data(), result.data(), resultBlob.size());

//         return {
//             .result = std::move(resultBlob)
//         };
//     }

//     virtual Dict runOp(std::string_view op, Dict params, ProgressCb) override {
//         auto ret = m_dispatcherData.dispatch(op, astl::move(params));
//         if (!ret) {
//             throw_ex{} << "tortoise: unknown op: " << op;
//         }
//         return *ret;
//     }
// };

// class TortoiseModel final : public Model {
//     std::shared_ptr<tortoise::Model> m_model;
// public:
//     using Schema = ac::schema::TortoiseProvider;

//     TortoiseModel(
//         std::string_view autoregressiveModelPath,
//         std::string_view diffusionModelPath,
//         std::string_view vocoderModelPath,
//         tortoise::Model::Params params)
//         : m_model(std::make_shared<tortoise::Model>(autoregressiveModelPath, diffusionModelPath, vocoderModelPath, astl::move(params)))
//     {}

//     virtual std::unique_ptr<Instance> createInstance(std::string_view, Dict params) override {
//         tortoise::Instance::InitParams initParams;
//         initParams.tokenizerPath = Dict_optValueAt(params, "tokenizerPath", std::string());
//         initParams.seed = Dict_optValueAt(params, "seed", 42);
//         return std::make_unique<TortoiseInstance>(m_model, astl::move(initParams));
//     }
// };

namespace sc = schema::tortoise;
using namespace ac::frameio;

struct BasicRunner {
    schema::OpDispatcherData m_dispatcherData;

    Frame dispatch(Frame& f) {
        try {
            auto ret = m_dispatcherData.dispatch(f.op, std::move(f.data));
            if (!ret) {
                throw_ex{} << "dummy: unknown op: " << f.op;
            }
            return {f.op, *ret};
        }
        catch (coro::IoClosed&) {
            throw;
        }
        catch (std::exception& e) {
            return {"error", e.what()};
        }
    }
};

SessionCoro<void> Tortoise_runInstance(coro::Io io, std::unique_ptr<tortoise::Instance> instance) {
    using Schema = sc::StateInstance;

    struct Runner : public BasicRunner {
        tortoise::Instance& m_instance;

        Runner(tortoise::Instance& instance)
            : m_instance(instance)
        {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        Schema::OpTTS::Return on(Schema::OpTTS, Schema::OpTTS::Params params) {
            const auto& text = params.text.value();
            const auto& voicePath = params.voicePath.value();

            auto result = m_instance.textToSpeech(text, voicePath);

            ac::Blob resultBlob;
            resultBlob.resize(result.size() * sizeof(float));
            memcpy(resultBlob.data(), result.data(), resultBlob.size());

            return {
                .audioData = std::move(resultBlob)
            };
        }
    };

    co_await io.pushFrame(Frame_stateChange(Schema::id));

    Runner runner(*instance);
    while (true) {
        auto f = co_await io.pollFrame();
        co_await io.pushFrame(runner.dispatch(f.frame));
    }
}

SessionCoro<void> Tortoise_runModel(coro::Io io, std::unique_ptr<tortoise::Model> model) {
    using Schema = sc::StateModelLoaded;

    struct Runner : public BasicRunner {
        Runner(tortoise::Model& model)
            : lmodel(model)
        {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        tortoise::Model& lmodel;
        std::unique_ptr<tortoise::Instance> instance;

        static tortoise::Instance::InitParams InstanceParams_fromSchema(sc::StateModelLoaded::OpStartInstance::Params& params) {
            tortoise::Instance::InitParams ret;
            ret.seed = params.seed.valueOr(42);
            ret.tokenizerPath = params.tokenizerPath.valueOr("");
            return ret;
        }

        Schema::OpStartInstance::Return on(Schema::OpStartInstance, Schema::OpStartInstance::Params params) {
            instance = std::make_unique<tortoise::Instance>(lmodel, InstanceParams_fromSchema(params));
            return {};
        }
    };

    co_await io.pushFrame(Frame_stateChange(Schema::id));

    Runner runner(*model);
    while (true) {
        auto f = co_await io.pollFrame();
        co_await io.pushFrame(runner.dispatch(f.frame));
        if (runner.instance) {
            co_await Tortoise_runInstance(io, std::move(runner.instance));
        }
    }
}

SessionCoro<void> Tortoise_runSession() {
    using Schema = sc::StateInitial;

    struct Runner : public BasicRunner {
        Runner() {
            schema::registerHandlers<Schema::Ops>(m_dispatcherData, *this);
        }

        std::unique_ptr<tortoise::Model> model;

        static tortoise::Model::Params ModelParams_fromSchema(sc::StateInitial::OpLoadModel::Params&) {
            tortoise::Model::Params ret;
            return ret;
        }

        Schema::OpLoadModel::Return on(Schema::OpLoadModel, Schema::OpLoadModel::Params params) {
            auto aggresiveModelPath = params.aggresivePath.valueOr("");
            auto diffusionModelPath = params.diffusionPath.valueOr("");
            auto vocoderModelPath = params.vocoderPath.valueOr("");
            auto lparams = ModelParams_fromSchema(params);

            model = std::make_unique<tortoise::Model>(
                aggresiveModelPath,
                diffusionModelPath,
                vocoderModelPath,
                astl::move(lparams));

            return {};
        }
    };

    try {
        auto io = co_await coro::Io{};

        co_await io.pushFrame(Frame_stateChange(Schema::id));

        Runner runner;

        while (true) {
            auto f = co_await io.pollFrame();
            co_await io.pushFrame(runner.dispatch(f.frame));
            if (runner.model) {
                co_await Tortoise_runModel(io, std::move(runner.model));
            }
        }
    }
    catch (coro::IoClosed&) {
        co_return;
    }
}

class TortoiseProvider final : public Provider {
public:
    virtual const Info& info() const noexcept override {
        static Info i = {
            .name = "ac tortoise.cpp",
            .vendor = "Alpaca Core"
        };
        return i;
    }

    /// Check if the model can be loaded
    // virtual bool canLoadModel(const ModelAssetDesc& desc, const Dict& /*params*/) const noexcept override {
    //      return desc.type == "tortoise.cpp";
    // }

    // virtual ModelPtr loadModel(ModelAssetDesc desc, Dict /*params*/, ProgressCb /*progressCb*/) override {
    //     if (desc.assets.size() != 3) throw_ex{} << "tortoise: expected exactly one local asset";
    //     auto& aggresive = desc.assets[0].path;
    //     auto& diffusion = desc.assets[1].path;
    //     auto& vocoder = desc.assets[2].path;
        // tortoise::Model::Params modelParams;
    //     return std::make_shared<TortoiseModel>(aggresive, diffusion, vocoder, modelParams);
    // }

    virtual frameio::SessionHandlerPtr createSessionHandler(std::string_view) override {
        return CoroSessionHandler::create(Tortoise_runSession());
    }
};
}

} // namespace ac::local

namespace ac::tortoise {

void init() {
    initLibrary();
}

std::vector<ac::local::ProviderPtr> getProviders() {
    std::vector<ac::local::ProviderPtr> ret;
    ret.push_back(std::make_unique<local::TortoiseProvider>());
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
        .getProviders = getProviders,
    };
}

} // namespace ac::tortoise

