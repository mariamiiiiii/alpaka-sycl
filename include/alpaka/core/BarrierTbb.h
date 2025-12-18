/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Comment this out to switch to tbb::task::suspend implementation. It utilizes sleep, instead of properly waiting
//#define ALPAKA_TBB_BARRIER_USE_MUTEX

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#    include "alpaka/core/Common.hpp"
#    include "alpaka/grid/Traits.hpp"

#    include <oneapi/tbb/task.h>
#    include <oneapi/tbb/task_group.h>
#    include <oneapi/tbb/concurrent_vector.h>
#    include <atomic>

#    include <iostream>
#    include <syncstream>
#    include <random>
#    include <thread>
#    include <chrono>

namespace alpaka::core
{
    namespace tbb
    {
        // A reusable barrier for TBB tasks using suspend/resume
        template<typename TIdx>
        class BarrierThread final
        {
        public:
            explicit BarrierThread(TIdx const& threadCount) : m_threadCount(threadCount) {
                suspended.reserve(threadCount);
            }

            // Called from inside a task to wait until all have arrived
            auto wait() -> void{
                oneapi::tbb::task::suspend(
                    [this](oneapi::tbb::task::suspend_point sp) mutable {
                        // push_back returns an iterator to the inserted element
                        auto it = suspended.push_back(std::move(sp));
                        TIdx count = std::distance(suspended.begin(), it) + 1;

                        std::osyncstream(std::cout)
                            /*<< "Task " << id*/ << " suspended (" << count << "/" << m_threadCount << ")\n";

                        if (count == m_threadCount) {
                            std::osyncstream(std::cout)
                                /*<< "Task " << id*/ << " is the last one — resuming all\n";

                            for (auto &sp2 : suspended) {
                                oneapi::tbb::task::resume(std::move(sp2));
                            }
                        }
                    });
            }

        private:
            TIdx const m_threadCount;
            oneapi::tbb::concurrent_vector<oneapi::tbb::task::suspend_point> suspended;
        };
    } // namespace tbb
} // namespace alpaka::core

#endif
