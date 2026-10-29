// /app/benchmark_rl_policies.cpp


#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include "aerial/rl/policy_evaluator.hpp"

namespace
{
    struct Options
    {
        std::size_t episodes = 10;
        std::uint32_t seed = 42;
        double maximum_episode_time = 20.0;
        std::string csv_path;
    };

    std::uint64_t ParseUnsigned(
        const std::string& value,
        const std::string& option)
    {
        if (value.empty() || value.front() == '-')
        {
            throw std::invalid_argument(
                "Invalid value for " + option
            );
        }

        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);

        if (consumed != value.size())
        {
            throw std::invalid_argument(
                "Invalid value for " + option
            );
        }

        return parsed;
    }

    Options ParseArguments(int argc, char** argv)
    {
        Options options;

        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];

            if (argument == "--help" || argument == "-h")
            {
                std::cout
                    << "Usage: benchmark_rl_policies [options]\n"
                    << "\nOptions:\n"
                    << "  --episodes N     Episodes per policy (default: 10)\n"
                    << "  --seed N         Initial seed (default: 42)\n"
                    << "  --duration S     Maximum episode duration in seconds\n"
                    << "                   (default: 20)\n"
                    << "  --csv PATH       Export per-episode results\n"
                    << "  --help           Show this message\n";

                std::exit(EXIT_SUCCESS);
            }

            if (i + 1 >= argc)
            {
                throw std::invalid_argument(
                    "Missing value for " + argument
                );
            }

            const std::string value = argv[++i];

            if (argument == "--episodes")
            {
                const auto parsed =
                    ParseUnsigned(value, argument);

                if (parsed == 0 ||
                    parsed > std::numeric_limits<std::size_t>::max())
                {
                    throw std::invalid_argument(
                        "Episode count must be positive"
                    );
                }

                options.episodes =
                    static_cast<std::size_t>(parsed);
            }
            else if (argument == "--seed")
            {
                const auto parsed =
                    ParseUnsigned(value, argument);

                if (parsed >
                    std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::invalid_argument(
                        "Seed exceeds uint32 range"
                    );
                }

                options.seed = static_cast<std::uint32_t>(parsed);
            }
            else if (argument == "--duration")
            {
                std::size_t consumed = 0;

                options.maximum_episode_time =
                    std::stod(value, &consumed);

                if (consumed != value.size() ||
                    !(options.maximum_episode_time > 0.0) ||
                    !std::isfinite(options.maximum_episode_time))
                {
                    throw std::invalid_argument(
                        "Duration must be finite and positive"
                    );
                }
            }
            else if (argument == "--csv")
            {
                if (value.empty())
                {
                    throw std::invalid_argument(
                        "CSV path cannot be empty"
                    );
                }

                options.csv_path = value;
            }
            else
            {
                throw std::invalid_argument(
                    "Unknown option: " + argument
                );
            }
        }

        return options;
    }

    const char* TerminationReasonName(
        aerial::rl::TerminationReason reason)
    {
        using aerial::rl::TerminationReason;

        switch (reason)
        {
            case TerminationReason::None:
                return "none";
            case TerminationReason::MinimumAltitude:
                return "minimum_altitude";
            case TerminationReason::MaximumAltitude:
                return "maximum_altitude";
            case TerminationReason::HorizontalBoundary:
                return "horizontal_boundary";
            case TerminationReason::ExcessiveTilt:
                return "excessive_tilt";
            case TerminationReason::NonFiniteState:
                return "non_finite_state";
        }

        return "unknown";
    }

    struct NamedResult
    {
        std::string name;
        aerial::rl::PolicyEvaluationResult evaluation;
    };

    void PrintSummary(
        const std::vector<NamedResult>& results)
    {
        std::cout << std::fixed << std::setprecision(3);

        std::cout
            << '\n'
            << std::left
            << std::setw(20) << "Policy"
            << std::right
            << std::setw(14) << "Mean reward"
            << std::setw(14) << "Mean time"
            << std::setw(16) << "Position err"
            << std::setw(14) << "Failure %"
            << '\n';

        std::cout << std::string(78, '-') << '\n';

        for (const auto& item : results)
        {
            const auto& result = item.evaluation;

            std::cout
                << std::left
                << std::setw(20) << item.name
                << std::right
                << std::setw(14) << result.mean_reward
                << std::setw(14) << result.mean_duration
                << std::setw(16) << result.mean_position_error
                << std::setw(14) << result.termination_rate * 100.0
                << '\n';
        }

        std::cout << '\n';

        for (const auto& item : results)
        {
            const auto& result = item.evaluation;

            std::cout
                << item.name << ":\n"
                << "  Reward range: "
                << result.minimum_reward << " to "
                << result.maximum_reward << '\n'
                << "  Mean velocity: "
                << result.mean_velocity << " m/s\n"
                << "  Truncation rate: "
                << result.truncation_rate * 100.0 << "%\n"
                << "  Altitude failures: "
                << result.minimum_altitude_failures +
                   result.maximum_altitude_failures << '\n'
                << "  Boundary failures: "
                << result.horizontal_boundary_failures << '\n'
                << "  Tilt failures: "
                << result.excessive_tilt_failures << '\n'
                << "  Non-finite failures: "
                << result.non_finite_failures << '\n';
        }
    }

    void ExportCSV(
        const std::string& path,
        const std::vector<NamedResult>& results,
        std::uint32_t base_seed)
    {
        std::ofstream file(path);

        if (!file)
        {
            throw std::runtime_error(
                "Failed to open CSV: " + path
            );
        }

        file << std::setprecision(17);

        file
            << "policy,episode,seed,steps,duration,"
            << "reward,mean_position_error,max_position_error,"
            << "mean_velocity,max_velocity,"
            << "terminated,truncated,termination_reason\n";

        for (const auto& item : results)
        {
            for (std::size_t i = 0;
                 i < item.evaluation.episodes.size();
                 ++i)
            {
                const auto& episode =
                    item.evaluation.episodes[i];

                file
                    << item.name << ','
                    << i << ','
                    << static_cast<std::uint64_t>(base_seed) + i << ','
                    << episode.steps << ','
                    << episode.duration << ','
                    << episode.cumulative_reward << ','
                    << episode.mean_position_error << ','
                    << episode.maximum_position_error << ','
                    << episode.mean_velocity << ','
                    << episode.maximum_velocity << ','
                    << episode.terminated << ','
                    << episode.truncated << ','
                    << TerminationReasonName(
                        episode.termination_reason
                    )
                    << '\n';
            }
        }

        if (!file)
        {
            throw std::runtime_error(
                "Failed while writing CSV: " + path
            );
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        const Options options = ParseArguments(argc, argv);

        aerial::rl::FlightEnvironmentParameters parameters;

        parameters.action_mode =
            aerial::rl::ActionMode::PIDTargets;

        parameters.maximum_episode_time =
            options.maximum_episode_time;

        aerial::rl::FlightEnvironment environment(parameters);

        aerial::rl::PolicyEvaluationParameters evaluation_parameters;
        evaluation_parameters.episode_count = options.episodes;
        evaluation_parameters.base_seed = options.seed;

        // PIDTargets action 0 requests the midpoint of each
        // configured target range. It does not mean zero altitude.
        aerial::rl::ConstantPolicy constant_policy({
            0.0, 0.0, 0.0, 0.0
        });

        aerial::rl::HoverFeedbackPolicyParameters feedback_parameters;

        feedback_parameters.target_altitude =
            parameters.hover_reward.target_position.z();

        feedback_parameters.maximum_horizontal_offset =
            parameters.pid_targets.maximum_horizontal_offset;

        feedback_parameters.minimum_altitude =
            parameters.pid_targets.minimum_altitude;

        feedback_parameters.maximum_altitude =
            parameters.pid_targets.maximum_altitude;

        feedback_parameters.observation_position_scale =
            parameters.observation_normalization.position_scale;

        aerial::rl::HoverFeedbackPolicy feedback_policy(
            feedback_parameters
        );

        std::vector<NamedResult> results;

        std::cout
            << "Project Aerial RL — Baseline Policy Benchmark\n"
            << "Episodes per policy: " << options.episodes << '\n'
            << "Base seed: " << options.seed << '\n'
            << "Episode limit: "
            << options.maximum_episode_time << " s\n"
            << "Action mode: PIDTargets\n";

        results.push_back({
            "Constant",
            aerial::rl::EvaluatePolicy(
                environment,
                constant_policy,
                evaluation_parameters
            )
        });

        results.push_back({
            "HoverFeedback",
            aerial::rl::EvaluatePolicy(
                environment,
                feedback_policy,
                evaluation_parameters
            )
        });

        PrintSummary(results);

        if (!options.csv_path.empty())
        {
            ExportCSV(
                options.csv_path,
                results,
                options.seed
            );

            std::cout
                << "\nCSV exported: "
                << options.csv_path << '\n';
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "Benchmark error: "
            << exception.what() << '\n';

        return EXIT_FAILURE;
    }
}

