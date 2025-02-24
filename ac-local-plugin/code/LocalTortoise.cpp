// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/tortoise/Instance.hpp>
#include <ac/tortoise/Init.hpp>
#include <ac/tortoise/Model.hpp>

#include <ac/local/Provider.hpp>
#include <ac/local/ProviderSessionContext.hpp>

#include <ac/schema/TortoiseCpp.hpp>
#include <ac/schema/OpDispatchHelpers.hpp>

#include <ac/FrameUtil.hpp>
#include <ac/frameio/IoEndpoint.hpp>

#include <ac/xec/coro.hpp>
#include <ac/io/exception.hpp>

#include <astl/move.hpp>
#include <astl/move_capture.hpp>
#include <astl/iile.h>
#include <astl/throw_stdex.hpp>
#include <astl/workarounds.h>

#include "aclp-tortoise-version.h"
#include "aclp-tortoise-interface.hpp"

namespace ac::local {

namespace {

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
        catch (io::stream_closed_error&) {
            throw;
        }
        catch (std::exception& e) {
            return {"error", e.what()};
        }
    }
};

xec::coro<void> Tortoise_runInstance(IoEndpoint& io, std::unique_ptr<tortoise::Instance> instance) {
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

    co_await io.push(Frame_stateChange(Schema::id));

    Runner runner(*instance);
    while (true) {
        auto f = co_await io.poll();
        co_await io.push(runner.dispatch(*f));
    }
}

xec::coro<void> Tortoise_runModel(IoEndpoint& io, std::unique_ptr<tortoise::Model> model) {
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

    co_await io.push(Frame_stateChange(Schema::id));

    Runner runner(*model);
    while (true) {
        auto f = co_await io.poll();
        co_await io.push(runner.dispatch(*f));
        if (runner.instance) {
            co_await Tortoise_runInstance(io, std::move(runner.instance));
        }
    }
}

xec::coro<void> Tortoise_runSession(StreamEndpoint ep) {
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
        auto ex = co_await xec::executor{};
        IoEndpoint io(std::move(ep), ex);

        co_await io.push(Frame_stateChange(Schema::id));

        Runner runner;

        while (true) {
            auto f = co_await io.poll();
            co_await io.push(runner.dispatch(*f));
            if (runner.model) {
                co_await Tortoise_runModel(io, std::move(runner.model));
            }
        }
    }
    catch (std::exception& e) {
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

    virtual void createSession(ProviderSessionContext ctx) override {
        co_spawn(ctx.executor.cpu, Tortoise_runSession(std::move(ctx.endpoint.session)));
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

