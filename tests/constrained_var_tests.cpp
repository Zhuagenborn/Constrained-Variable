#include "constrained_var/constrained_var.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace cv;
using namespace cv::opt;
using namespace cv::opt::chk;

namespace {
enum class Color { White, Red, Green, Black, Invalid };
}

template <>
struct cv::opt::EnumValues<Color> {
    static constexpr std::array values {Color::White, Color::Red, Color::Green, Color::Black};
};

namespace {

inline constexpr int min {10}, max {100}, low {min + 10}, high {max - 10};

ChainType<std::vector<int>> SizeToVector(const std::size_t& size) noexcept {
    return std::vector<int>(size);
}

}  // namespace

TEST(Constraint, IsOpt) {
    EXPECT_FALSE(IsOpt<int>);
    EXPECT_TRUE(IsOpt<Min<int>>);
}

TEST(Constraint, IsSameOpt) {
    EXPECT_FALSE((IsSameOpt<int, OptType::Min>));
    EXPECT_TRUE((IsSameOpt<Min<int>, OptType::Min>));
}

TEST(Constraint, IsValidOptChain) {
    EXPECT_TRUE(IsValidOptChain<int>);
    EXPECT_TRUE((IsValidOptChain<int, Min<int>>));
    EXPECT_FALSE((IsValidOptChain<int, Min<std::string>>));

    EXPECT_TRUE((IsValidOptChain<int, Transformer<int, std::string>, InSet<std::string>>));
    EXPECT_FALSE((IsValidOptChain<int, Transformer<int, std::string>, InSet<int>>));
}

TEST(Constraint, IsEmptyOptChain) {
    EXPECT_TRUE(IsEmptyOptChain<>);
    EXPECT_FALSE(IsEmptyOptChain<int>);
    EXPECT_FALSE((IsEmptyOptChain<int, Min<int>>));
}

TEST(ConstrainedVariable, MinMax) {
    constexpr Min<int> min_opt {min};
    constexpr Max<int> max_opt {max};

    {
        Variable<int> var;
        EXPECT_EQ(var.Set(min - 1).value_or(min), min - 1);
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        MinVariable<int> var {min_opt};
        EXPECT_FALSE(var.Set(min - 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        MaxVariable<int> var {max_opt};
        EXPECT_FALSE(var.Set(max + 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
    {
        ConstrainedVariable<int, decltype(min_opt), decltype(max_opt)> var {min_opt, max_opt};
        EXPECT_FALSE(var.Set(min - 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_FALSE(var.Set(max + 1).has_value());
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
}

TEST(ConstrainedVariable, NotInRange) {
    constexpr NotInRange<int> not_in_range_opt {{min, max}};

    {
        Variable<int> var;
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
    {
        NotInRangeVariable<int> var {not_in_range_opt};
        EXPECT_FALSE(var.Set(min + 1).has_value());
        EXPECT_EQ(var.Set(min - 1).value_or(min), min - 1);
        EXPECT_FALSE(var.Set(max - 1).has_value());
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        NotInRangeVariable<int> inclusive_var {not_in_range_opt};
        EXPECT_FALSE(inclusive_var.Set(min).has_value());
        EXPECT_FALSE(inclusive_var.Set(max).has_value());

        constexpr NotInRange<int> exclusive_not_in_range_opt {{min, max},
                                                              {BoundType::Open, BoundType::Open}};
        NotInRangeVariable<int> exclusive_var {exclusive_not_in_range_opt};
        EXPECT_TRUE(exclusive_var.Set(min).has_value());
        EXPECT_TRUE(exclusive_var.Set(max).has_value());
    }
}

TEST(ConstrainedVariable, InRange) {
    constexpr InRange<int> in_range_opt {{min, max}};

    {
        Variable<int> var;
        EXPECT_EQ(var.Set(min - 1).value_or(min), min - 1);
        EXPECT_EQ(var.Set(max + 1).value_or(max), max + 1);
    }
    {
        InRangeVariable<int> var {in_range_opt};
        EXPECT_FALSE(var.Set(min - 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), min + 1);
        EXPECT_FALSE(var.Set(max + 1).has_value());
        EXPECT_EQ(var.Set(max - 1).value_or(max), max - 1);
    }
    {
        InRangeVariable<int> inclusive_var {in_range_opt};
        EXPECT_TRUE(inclusive_var.Set(min).has_value());
        EXPECT_TRUE(inclusive_var.Set(max).has_value());

        constexpr InRange<int> exclusive_in_range_opt {{min, max},
                                                       {BoundType::Open, BoundType::Open}};
        InRangeVariable<int> exclusive_var {exclusive_in_range_opt};
        EXPECT_FALSE(exclusive_var.Set(min).has_value());
        EXPECT_FALSE(exclusive_var.Set(max).has_value());
    }
}

TEST(ConstrainedVariable, Enum) {
    {
        Variable<Color> var;
        EXPECT_EQ(var.Set(Color::Invalid).value_or(Color::Black), Color::Invalid);
        EXPECT_EQ(var.Set(Color::White).value_or(Color::Black), Color::White);
    }
    {
        EnumVariable<Color> var;
        EXPECT_FALSE(var.Set(Color::Invalid).has_value());
        EXPECT_EQ(var.Set(Color::White).value_or(Color::Black), Color::White);
    }
}

TEST(ConstrainedVariable, Clamp) {
    constexpr Clamp<int> clamp_opt {{low, high}};

    {
        Variable<int> var;
        EXPECT_EQ(var.Set(low - 1).value_or(low), low - 1);
        EXPECT_EQ(var.Set(high + 1).value_or(high), high + 1);
    }
    {
        ClampVariable<int> var {clamp_opt};
        EXPECT_EQ(var.Set(low - 1).value_or(0), low);
        EXPECT_EQ(var.Set(high + 1).value_or(0), high);
    }
    {
        constexpr Min<int> min_opt {max};
        ConstrainedVariable<int, decltype(clamp_opt), decltype(min_opt)> var {clamp_opt, min_opt};
        EXPECT_FALSE(var.Set(0).has_value());
        EXPECT_FALSE(var.Set(min + 1).has_value());
        EXPECT_FALSE(var.Set(max - 1).has_value());
        EXPECT_FALSE(var.Set(low + 1).has_value());
        EXPECT_FALSE(var.Set(high - 1).has_value());
    }
    {
        constexpr InRange<int> in_range_opt {{min, max}};
        ConstrainedVariable<int, decltype(in_range_opt), decltype(clamp_opt)> var {in_range_opt,
                                                                                   clamp_opt};
        EXPECT_FALSE(var.Set(min - 1).has_value());
        EXPECT_EQ(var.Set(min + 1).value_or(min), low);

        EXPECT_FALSE(var.Set(max + 1).has_value());
        EXPECT_EQ(var.Set(max - 1).value_or(max), high);

        EXPECT_EQ(var.Set(low - 1).value_or(0), low);
        EXPECT_EQ(var.Set(high + 1).value_or(0), high);
    }
}

TEST(ConstrainedVariable, NotEmpty) {
    const std::vector<int> empty_vals;

    {
        Variable<std::vector<int>> var;
        EXPECT_EQ(var.Set(empty_vals).value_or(std::vector<int> {1}), empty_vals);
    }
    {
        NotEmptyVariable<std::vector<int>> var;
        EXPECT_FALSE(var.Set(empty_vals).has_value());
        EXPECT_EQ(var.Set(std::vector<int> {1}).value_or(empty_vals), std::vector<int> {1});
    }
}

TEST(ConstrainedVariable, Predicate) {
    const Predicate<int> predicate_opt {[](const int& val) noexcept -> bool {
        return val >= 0;
    }};

    PredicateVariable<int> var {predicate_opt};
    EXPECT_EQ(var.Set(1).value_or(0), 1);
    EXPECT_FALSE(var.Set(-1).has_value());
}

TEST(ConstrainedVariable, InSet) {
    const InSet<int> in_set_opt {1, 2};
    InSetVariable<int> var {in_set_opt};
    EXPECT_EQ(var.Set(1).value_or(0), 1);
    EXPECT_EQ(var.Set(2).value_or(0), 2);
    EXPECT_FALSE(var.Set(3).has_value());
}

TEST(ConstrainedVariable, NotInSet) {
    const NotInSet<int> not_in_set_opt {1, 2};
    NotInSetVariable<int> var {not_in_set_opt};
    EXPECT_FALSE(var.Set(1).has_value());
    EXPECT_FALSE(var.Set(2).has_value());
    EXPECT_EQ(var.Set(3).value_or(0), 3);
}

TEST(ConstrainedVariable, NotNullOpt) {
    {
        Variable<int> var;
        EXPECT_EQ(var.Set(1).value_or(0), 1);
        EXPECT_EQ(var.Set(0).value_or(1), 0);
    }
    {
        NotNullVariable<int> var;
        EXPECT_EQ(var.Set(1).value_or(0), 1);
        EXPECT_FALSE(var.Set(0).has_value());
    }
}

TEST(ConstrainedVariable, Transformer) {
    constexpr NotEmpty<std::vector<int>> not_empty_opt;
    const Transformer<std::size_t, std::vector<int>> transformer_opt {SizeToVector};

    {
        TransformerVariable<std::size_t, std::vector<int>> var {transformer_opt};
        EXPECT_EQ(var.Set(0).value_or(std::vector<int>(1)), std::vector<int> {});
        EXPECT_EQ(var.Set(1).value_or(std::vector<int> {}), std::vector<int>(1));
    }
    {
        ConstrainedVariable<std::vector<int>, decltype(transformer_opt), decltype(not_empty_opt)>
            var {transformer_opt, not_empty_opt};
        EXPECT_FALSE(var.Set(0).has_value());
        EXPECT_EQ(var.Set(1).value_or(std::vector<int> {}), std::vector<int>(1));
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
        ValidationChain<int, decltype(min_opt), decltype(max_opt)> var {min_opt, max_opt};
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

TEST(ValidatedBoolVariable, Set) {
    {
        const InSet<int> in_set_opt {1, 2};
        ValidatedBoolVariable<int, decltype(in_set_opt)> var {in_set_opt};
        EXPECT_TRUE(var.Set(1).value());
        EXPECT_TRUE(var.Set(2).value());
        EXPECT_FALSE(var.Set(3).value());
    }
    {
        constexpr Min<int> min_opt {min};
        constexpr Max<int> max_opt {max};
        ValidatedBoolVariable<int, decltype(min_opt), decltype(max_opt)> var {min_opt, max_opt};
        EXPECT_FALSE(var.Set(min - 1).value());
        EXPECT_TRUE(var.Set(min + 1).value());
        EXPECT_FALSE(var.Set(max + 1).value());
        EXPECT_TRUE(var.Set(max - 1).value());
    }
    {
        const InSet<std::string> in_set_opt {"name", "age"};
        ValidatedBoolVariable<std::string, decltype(in_set_opt)> var {in_set_opt};
        EXPECT_TRUE(var.Set("age").value());
        EXPECT_TRUE(var.Set("name").value());
        EXPECT_FALSE(var.Set("gender").value());
    }
}