// /apps/benchmark_imitation_learning.cpp

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <string>

#include "aerial/rl/adam_optimizer.hpp"
#include "aerial/rl/epoch_trainer.hpp"
#include "aerial/rl/imitation_dataset_collector.hpp"
#include "aerial/rl/neural_network_policy.hpp"
#include "aerial/rl/policy_evaluator.hpp"

namespace
{
    struct Options
    {
        std::size_t episodes = 40;
        std::size_t epochs = 100;
        std::size_t batch_size = 64;

        std::uint32_t seed = 42;

        double duration = 5.0;
        double validation_fraction = 0.2;

        std::string checkpoint = "benchmarks/results/imitation_policy.bin";
    };

    std::size_t ParsePositiveSize(const std::string& value)
    {
        std::size_t position = 0;
        const auto result = std::stoull(value, &position);

        if (position != value.size() || result == 0 ||
            result > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("Invalid positive integer");
        }

        return static_cast<std::size_t>(result);
    }

    std::uint32_t ParseSeed(const std::string& value)
    {
        std::size_t position = 0;
        const auto result = std::stoull(value, &position);

        if (position != value.size() ||
            result > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("Invalid seed");
        }

        return static_cast<std::uint32_t>(result);
    }

    double ParsePositiveDouble(const std::string& value)
    {
        std::size_t position = 0;
        const double result = std::stod(value, &position);
        
        if (position != value.size() ||
            !std::isfinite(result) || result <= 0.0) {
            throw std::invalid_argument("Invalid positive number");
        }

        return result;
    }

    Options ParseArguments(int argc, char** argv)
    {
        Options options;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];

            if (argument == "--help") {
                std::cout
                    << "Options:\n"
                    << "  --episodes N\n"
                    << "  --epochs N\n"
                    << "  --batch-size N\n"
                    << "  --seed N\n"
                    << "  --duration SECONDS\n"
                    << "  --validation FRACTION\n"
                    << "  --checkpoint PATH\n";
                std::exit(0);
            }

            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + argument);
            }

            const std::string value = argv[++i];

            if (argument == "--episodes")
                options.episodes = ParsePositiveSize(value);
            else if (argument == "--epochs")
                options.epochs = ParsePositiveSize(value);
            else if (argument == "--batch-size")
                options.batch_size = ParsePositiveSize(value);
            else if (argument == "--seed")
                options.seed = ParseSeed(value);
            else if (argument == "--duration")
                options.duration = ParsePositiveDouble(value);
            else if (argument == "--validation")
                options.validation_fraction =
                    ParsePositiveDouble(value);
            else if (argument == "--checkpoint")
                options.checkpoint = value;
            else
                throw std::invalid_argument("Unknown argument: " + argument);
        }

        if (options.episodes < 2)
            throw std::invalid_argument(
                "At least two episodes required"
            );

        if (options.validation_fraction >= 1.0)
            throw std::invalid_argument(
                "Validation fraction must be less than 1"
            );

        if (options.checkpoint.empty())
            throw std::invalid_argument(
                "Checkpoint path must not be empty"
            );

        if (options.episodes >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max() -
                options.seed
            ) + 1)
        {
            throw std::invalid_argument(
                "Episode seeds would overflow"
            );
        }

        return options;
    }

    void PrintEvaluation(
            const std::string& name,
            const aerial::rl::PolicyEvaluationResult& result
            )
    {
        std::cout
            << "\n" << name << '\n'
            << "  Mean reward:         "
            << result.mean_reward << '\n'
            << "  Mean position error: "
            << result.mean_position_error << " m\n"
            << "  Mean velocity:       "
            << result.mean_velocity << " m/s\n"
            << "  Termination rate:    "
            << result.termination_rate * 100.0 << "%\n"
            << "  Truncation rate:     "
            << result.truncation_rate * 100.0 << "%\n";
    }
}

int main(int argc, char** argv)
{
    using namespace aerial::rl;

    try
    {
        const Options options = ParseArguments(argc, argv);

        FlightEnvironmentParameters environment_parameters;

        environment_parameters.action_mode = ActionMode::PIDTargets;

        environment_parameters.maximum_episode_time = options.duration;

        FlightEnvironment environment(environment_parameters);

        HoverFeedbackPolicy teacher;

        ImitationCollectionParameters collection_parameters;
        collection_parameters.episode_count = options.episodes;
        collection_parameters.base_seed = options.seed;

        std::cout << "Collecting teacher demonstrations...\n";

        const auto collection = CollectImitationDataset(
                environment,
                teacher,
                collection_parameters
                );

        const auto split = SplitImitationEpisodes(
                collection,
                options.validation_fraction,
                options.seed
                );

        std::cout
            << "Episodes:           "
            << collection.episodes.size() << '\n'
            << "Total samples:      "
            << collection.dataset.Size() << '\n'
            << "Training samples:   "
            << split.training_indices.size() << '\n'
            << "Validation samples: "
            << split.validation_indices.size() << '\n';

        NeuralNetworkPolicy student(options.seed);

        AdamOptimizer optimizer(student.GetParameterCount());

        const double initial_training_loss = EvaluateDatasetMSE(
                student,
                collection.dataset,
                split.training_indices
                );

        const double initial_validation_loss = EvaluateDatasetMSE(
                student,
                collection.dataset,
                split.validation_indices
                );

        std::cout << std::fixed << std::setprecision(8)
                  << "\nInitial training MSE:   "
                  << initial_training_loss
                  << "\nInitial validation MSE: "
                  << initial_validation_loss << '\n';

        double best_validation_loss = std::numeric_limits<double>::infinity();

        std::size_t best_epoch = 0;

        // Preserve the best network, not merely the final epoch
        auto best_parameters = student.GetParameters();

        for (std::size_t epoch = 0; epoch < options.epochs; ++ epoch) {
            const auto metrics = TrainEpoch(
                    student,
                    optimizer,
                    collection.dataset,
                    split,
                    options.batch_size,
                    options.seed + static_cast<std::uint32_t>(
                        epoch % (
                            static_cast<std::size_t>(
                                std::numeric_limits<std::uint32_t>::max()
                                ) + 1
                            )
                        )
                    );

            if (metrics.validation_loss < best_validation_loss) {
                best_validation_loss = metrics.validation_loss;
                best_epoch = epoch + 1;
                best_parameters = student.GetParameters();
            }

            if ((epoch + 1) % 10 == 0 ||
                epoch == 0 ||
                epoch + 1 == options.epochs) {
                std::cout
                    << "Epoch " << std::setw(4) << epoch + 1
                    << " | Train MSE: "
                    << metrics.training_loss
                    << " | Val MSE: "
                    << metrics.validation_loss
                    << '\n';
            }
        }

        student.SetParameters(best_parameters);

        std::cout
            << "\nBest epoch:          " << best_epoch
            << "\nBest validation MSE: "
            << best_validation_loss << '\n';

        const std::filesystem::path checkpoint_path(
                options.checkpoint
                );

        if (!checkpoint_path.parent_path().empty()) {
            std::filesystem::create_directories(
                    checkpoint_path.parent_path()
                    );
        }

        student.SaveParameters(options.checkpoint);

        std::cout
            << "Saved checkpoint: "
            << options.checkpoint << '\n';

        // Use seeds that were not part of demonstration collection
        if (options.episodes >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max() -
                options.seed - 1000U
                )) {
            throw std::invalid_argument("Not enough seed space for separate evaluation");
        }

        PolicyEvaluationParameters evaluation_parameters;
        evaluation_parameters.episode_count = 20;
        evaluation_parameters.base_seed =
            options.seed +
            static_cast<std::uint32_t>(options.episodes) +
            1000U;

        const auto teacher_results = EvaluatePolicy(
                environment,
                teacher,
                evaluation_parameters
                );

        const auto student_results = EvaluatePolicy(
                environment,
                student,
                evaluation_parameters
                );

        PrintEvaluation("Teacher controller", teacher_results);
        PrintEvaluation("Neural network student", student_results);

        std::cout
            << "\nImitation-learning benchmark complete.\n";

        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << '\n';

        return 1;
    }
}
