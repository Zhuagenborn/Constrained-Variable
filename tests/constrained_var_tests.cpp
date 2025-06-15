#include "constrained_var/constrained_var.h"

#include <gtest/gtest.h>

#include <vector>

using namespace cv;
using namespace cv::opt;
using namespace cv::opt::chk;

namespace {
enum class Color { White, Red, Green, Black, Invalid };
}

template <>
struct cv::opt::EnumValues<Color> {
    static constexpr std::array values {Color::White, Color::Red, Color::Green,
                                        Color::Black};
};

namespace {

template <typename T, IsOpt... Opts>
using CV = ConstrainedVariable<T, Opts...>;

inline constexpr int min {10}, max {100}, low {min + 10}, high {max - 10};

ChainType<std::vector<int>> SizeToVector(const std::size_t& size) noexcept {
    return std::vector<int>(size);
}

}  // namespace

TEST(ConstrainedVariable, IsOpt) {
    using MinOpt = Min<int>;

    EXPECT_FALSE(IsOpt<int>);
    EXPECT_TRUE(IsOpt<MinOpt>);
}

TEST(ConstrainedVariable, IsSameOpt) {
    using MinOpt = Min<int>;

    EXPECT_FALSE((IsSameOpt<int, OptType::Min>));
    EXPECT_TRUE((IsSameOpt<MinOpt, OptType::Min>));
}

TEST(ConstrainedVariable, MinMax) {
    constexpr Min<int> min_opt {min};
    constexpr Max<int> max_opt {max};

    {
        CV<int> var;
        EXPECT_EQ(var.Set(min - 1).value_or(min), min - 1);
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        CV<int, decltype(min_opt)> var {min_opt};
        EXPECT_FALSE(var.Set(min - 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        CV<int, decltype(max_opt)> var {max_opt};
        EXPECT_FALSE(var.Set(max + 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
    {
        CV<int, decltype(min_opt), decltype(max_opt)> var {min_opt, max_opt};
        EXPECT_FALSE(var.Set(min - 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_FALSE(var.Set(max + 1).has_value());
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
}

TEST(ConstrainedVariable, NotInRange) {
    constexpr NotInRange<int> not_in_range_opt {{min, max}};

    {
        CV<int> var;
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
    {
        CV<int, decltype(not_in_range_opt)> var {not_in_range_opt};
        EXPECT_FALSE(var.Set(min + 1).has_value());
        EXPECT_EQ(var.Set(min - 1).value_or(min), min - 1);
        EXPECT_FALSE(var.Set(max - 1).has_value());
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        CV<int, decltype(not_in_range_opt)> inclusive_var {not_in_range_opt};
        EXPECT_FALSE(inclusive_var.Set(min).has_value());
        EXPECT_FALSE(inclusive_var.Set(max).has_value());

        constexpr NotInRange<int> exclusive_not_in_range_opt {
            {min, max}, {BoundType::Open, BoundType::Open}};
        CV<int, decltype(exclusive_not_in_range_opt)> exclusive_var {
            exclusive_not_in_range_opt};
        EXPECT_TRUE(exclusive_var.Set(min).has_value());
        EXPECT_TRUE(exclusive_var.Set(max).has_value());
    }
}

TEST(ConstrainedVariable, InRange) {
    constexpr InRange<int> in_range_opt {{min, max}};

    {
        CV<int> var;
        EXPECT_EQ(var.Set(min - 1).value_or(min), min - 1);
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        CV<int, decltype(in_range_opt)> var {in_range_opt};
        EXPECT_FALSE(var.Set(min - 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_FALSE(var.Set(max + 1).has_value());
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
    {
        CV<int, decltype(in_range_opt)> inclusive_var {in_range_opt};
        EXPECT_TRUE(inclusive_var.Set(min).has_value());
        EXPECT_TRUE(inclusive_var.Set(max).has_value());

        constexpr InRange<int> exclusive_in_range_opt {
            {min, max}, {BoundType::Open, BoundType::Open}};
        CV<int, decltype(exclusive_in_range_opt)> exclusive_var {
            exclusive_in_range_opt};
        EXPECT_FALSE(exclusive_var.Set(min).has_value());
        EXPECT_FALSE(exclusive_var.Set(max).has_value());
    }
}

TEST(ConstrainedVariable, Enum) {
    constexpr Enum<Color> enum_opt;

    {
        CV<Color> var;
        EXPECT_EQ(var.Set(Color::Invalid).value_or(Color::Black),
                  Color::Invalid);
        EXPECT_EQ(var.Set(Color::White).value_or(Color::Black), Color::White);
    }
    {
        CV<Color, decltype(enum_opt)> var {enum_opt};
        EXPECT_FALSE(var.Set(Color::Invalid).has_value());
        EXPECT_EQ(var.Set(Color::White).value_or(Color::Black), Color::White);
    }
}

TEST(ConstrainedVariable, Clamp) {
    constexpr Clamp<int> clamp_opt {{low, high}};

    {
        CV<int> var;
        EXPECT_EQ(var.Set(low - 1).value_or(low), low - 1);
        EXPECT_EQ(var.Set(high + 1).value_or(high), high + 1);
    }
    {
        CV<int, decltype(clamp_opt)> var {clamp_opt};
        EXPECT_EQ(var.Set(low - 1).value_or(0), low);
        EXPECT_EQ(var.Set(high + 1).value_or(0), high);
    }
    {
        constexpr Min<int> min_opt {max};
        CV<int, decltype(clamp_opt), decltype(min_opt)> var {clamp_opt,
                                                             min_opt};
        EXPECT_FALSE(var.Set(0).has_value());
        EXPECT_FALSE(var.Set(min + 1).has_value());
        EXPECT_FALSE(var.Set(max - 1).has_value());
        EXPECT_FALSE(var.Set(low + 1).has_value());
        EXPECT_FALSE(var.Set(high - 1).has_value());
    }
}

TEST(ConstrainedVariable, InRangeThenClamp) {
    constexpr Clamp<int> clamp_opt {{low, high}};
    constexpr InRange<int> in_range_opt {{min, max}};

    CV<int, decltype(in_range_opt), decltype(clamp_opt)> var {in_range_opt,
                                                              clamp_opt};

    EXPECT_FALSE(var.Set(min - 1).has_value());
    EXPECT_EQ(var.Set(min + 1).value_or(min), low);

    EXPECT_FALSE(var.Set(max + 1).has_value());
    EXPECT_EQ(var.Set(max - 1).value_or(max), high);

    EXPECT_EQ(var.Set(low - 1).value_or(0), low);
    EXPECT_EQ(var.Set(high + 1).value_or(0), high);
}

TEST(ConstrainedVariable, NotEmpty) {
    constexpr NotEmpty<std::vector<int>> not_empty_opt;
    const std::vector<int> empty_vals;

    {
        CV<std::vector<int>> var;
        EXPECT_EQ(var.Set(empty_vals).value_or(std::vector<int> {1}),
                  empty_vals);
    }
    {
        CV<std::vector<int>, decltype(not_empty_opt)> var {not_empty_opt};
        EXPECT_FALSE(var.Set(empty_vals).has_value());
        EXPECT_EQ(var.Set(std::vector<int> {1}).value_or(empty_vals),
                  std::vector<int> {1});
    }
}

TEST(ConstrainedVariable, Predicate) {
    const Predicate<int> predicate_opt {[](const int& val) noexcept -> bool {
        return val >= 0;
    }};

    CV<int, decltype(predicate_opt)> var {predicate_opt};
    EXPECT_EQ(var.Set(1).value_or(0), 1);
    EXPECT_FALSE(var.Set(-1).has_value());
}

TEST(ConstrainedVariable, InSet) {
    const InSet<int> in_set_opt {1, 2};

    CV<int, decltype(in_set_opt)> var {in_set_opt};
    EXPECT_EQ(var.Set(1).value_or(0), 1);
    EXPECT_EQ(var.Set(2).value_or(0), 2);
    EXPECT_FALSE(var.Set(3).has_value());
}

TEST(ConstrainedVariable, NotInSet) {
    const NotInSet<int> not_in_set_opt {1, 2};

    CV<int, decltype(not_in_set_opt)> var {not_in_set_opt};
    EXPECT_FALSE(var.Set(1).has_value());
    EXPECT_FALSE(var.Set(2).has_value());
    EXPECT_EQ(var.Set(3).value_or(0), 3);
}

TEST(ConstrainedVariable, NotNullOpt) {
    constexpr NotNull<int> not_null_opt;

    {
        CV<int> var;
        EXPECT_EQ(var.Set(0).value_or(1), 0);
    }
    {
        CV<int, decltype(not_null_opt)> var {not_null_opt};
        EXPECT_FALSE(var.Set(0).has_value());
    }
}

TEST(ConstrainedVariable, ValidSetToBool) {
    const InSet<int> in_set_opt {1, 2};
    const Transformer<ChainType<int>, bool> transformer_opt {
        [](const ChainType<int>& val) noexcept {
            return val.has_value();
        }};

    CV<bool, decltype(in_set_opt), decltype(transformer_opt)> var {
        in_set_opt, transformer_opt};
    EXPECT_TRUE(var.Set(1).has_value());
    EXPECT_TRUE(var.Get());

    EXPECT_TRUE(var.Set(2).has_value());
    EXPECT_TRUE(var.Get());

    EXPECT_TRUE(var.Set(3).has_value());
    EXPECT_FALSE(var.Get());
}

TEST(ConstrainedVariable, Transformer) {
    constexpr NotEmpty<std::vector<int>> not_empty_opt;
    const Transformer<std::size_t, std::vector<int>> transformer_opt {
        SizeToVector};

    {
        CV<std::vector<int>, decltype(transformer_opt)> var {transformer_opt};
        EXPECT_EQ(var.Set(0).value_or(std::vector<int>(1)),
                  std::vector<int> {});
        EXPECT_EQ(var.Set(1).value_or(std::vector<int> {}),
                  std::vector<int>(1));
    }
    {
        CV<std::vector<int>, decltype(transformer_opt), decltype(not_empty_opt)>
            var {transformer_opt, not_empty_opt};
        EXPECT_FALSE(var.Set(0).has_value());
        EXPECT_EQ(var.Set(1).value_or(std::vector<int> {}),
                  std::vector<int>(1));
    }
}

TEST(ValidationChain, Apply) {
    {
        const InSet<int> in_set_opt {1, 2};
        ValidationChain<int, decltype(in_set_opt)> var {in_set_opt};
        EXPECT_TRUE(var.Apply(1));
        EXPECT_TRUE(var.Apply(2));
        EXPECT_FALSE(var.Apply(3));
    }
    {
        constexpr Min<int> min_opt {min};
        constexpr Max<int> max_opt {max};
        ValidationChain<int, decltype(min_opt), decltype(max_opt)> var {
            min_opt, max_opt};
        EXPECT_FALSE(var.Apply(min - 1));
        EXPECT_TRUE(var.Apply(min + 1));
        EXPECT_FALSE(var.Apply(max + 1));
        EXPECT_TRUE(var.Apply(max - 1));
    }
    {
        const InSet<std::string> in_set_opt {"name", "age"};
        ValidationChain<std::string, decltype(in_set_opt)> var {in_set_opt};
        EXPECT_TRUE(var.Apply("age"));
        EXPECT_TRUE(var.Apply("name"));
        EXPECT_FALSE(var.Apply("gender"));
    }
}

TEST(ValidBoolVariable, Set) {
    {
        const InSet<int> in_set_opt {1, 2};
        ValidBoolVariable<int, decltype(in_set_opt)> var {in_set_opt};
        EXPECT_TRUE(var.Set(1));
        EXPECT_TRUE(var.Set(2));
        EXPECT_FALSE(var.Set(3));
    }
    {
        constexpr Min<int> min_opt {min};
        constexpr Max<int> max_opt {max};
        ValidBoolVariable<int, decltype(min_opt), decltype(max_opt)> var {
            min_opt, max_opt};
        EXPECT_FALSE(var.Set(min - 1));
        EXPECT_TRUE(var.Set(min + 1));
        EXPECT_FALSE(var.Set(max + 1));
        EXPECT_TRUE(var.Set(max - 1));
    }
    {
        const InSet<std::string> in_set_opt {"name", "age"};
        ValidBoolVariable<std::string, decltype(in_set_opt)> var {in_set_opt};
        EXPECT_TRUE(var.Set("age"));
        EXPECT_TRUE(var.Set("name"));
        EXPECT_FALSE(var.Set("gender"));
    }
}