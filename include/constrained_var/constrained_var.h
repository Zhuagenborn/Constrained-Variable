/**
 * @file constrained_var.h
 * @brief Constraint-based validation utilities for variables.
 *
 * @details
 * A header-only library written in *C++23* for applying composable validation and transformation constraints
 * to variables in a flexible and type-safe way, supporting:
 *
 * - Range and set checks.
 * - Null and emptiness checks.
 * - Enumeration restrictions.
 * - Value transformation.
 * - Custom predicate validation.
 *
 * Constraints are applied in a user-defined order and ensure the values are within the expected bounds or rules.
 * Helpful error messages are generated when constraints are violated.
 *
 * @par GitHub
 * https://github.com/Zhuagenborn
 * @version 1.0
 * @date 2025-06-13
 *
 * @example tests/constrained_var_tests.cpp
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <ranges>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace cv {

//! The error with an error code and a human-readable message.
using Error = std::pair<std::error_code, std::string>;

//! The argument and return type in constraint set chain.
template <typename T>
using ChainType = std::expected<T, Error>;

//! The namespace for constraints.
namespace opt {

//! Constraint types.
enum class OptType {
    Min,
    Max,
    Clamp,
    InRange,
    NotInRange,
    NotNull,
    NotEmpty,
    Transformer,
    Enum,
    Predicate,
    InSet,
    NotInSet
};

//! Whether a type can be formatted by @p std::format.
template <typename T>
concept FormatInvocable = requires(
    const T& val, std::formatter<T, char> formatter, std::format_context& ctx) {
    { formatter.format(val, ctx) } -> std::same_as<decltype(ctx.out())>;
};

//! The bound type for ranges.
enum class BoundType {
    //! [min, max]
    Closed,
    //! (min, max)
    Open
};

constexpr std::pair<char, char> BuildBoundBrackets(
    const std::pair<BoundType, BoundType> bounds) noexcept {
    const auto left {bounds.first == BoundType::Closed ? '[' : '('};
    const auto right {bounds.second == BoundType::Closed ? ']' : ')'};
    return {left, right};
}

//! Specialize this template to define valid values for scoped enumeration types.
template <typename T>
    requires std::is_scoped_enum_v<T>
class EnumValues {
public:
    static constexpr std::array<T, 0> values {};
};

/**
 * @brief The constraint for checking if an enumeration value is within allowed range.
 *
 * @note The enumeration type must specialize @ref EnumValues.
 */
template <typename T>
    requires std::is_scoped_enum_v<T>
class Enum {
public:
    static constexpr OptType type {OptType::Enum};

    /**
     * @brief Apply the constraint to a given value.
     *
     * @param var The value to validate, possibly the result of a previous constraint in the chain.
     * @return
     * - If @ref var has a value:
     *   - Return the original value if it satisfies the constraint.
     *   - Otherwise, return an unexpected error.
     * - Otherwise, the constraint is skipped and return the unchanged error.
     */
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (const auto raw_val {std::to_underlying(val)};
                raw_val < min || raw_val > max) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::result_out_of_range),
                     BuildErrorMsg(raw_val)}};
            } else {
                return val;
            }
        });
    }

private:
    std::string BuildErrorMsg(
        const std::underlying_type_t<T> val) const noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format("The enumeration {} is out of the allowed "
                               "range [{}, {}]",
                               val, min, max);
        } else {
            return "The enumeration is out of the allowed range";
        }
    }

    static constexpr auto& vals {EnumValues<T>::values};
    static_assert(!vals.empty());

    static constexpr std::underlying_type_t<T> min {
        std::to_underlying(*std::ranges::min_element(vals))};
    static constexpr std::underlying_type_t<T> max {
        std::to_underlying(*std::ranges::max_element(vals))};
    static_assert(min <= max);
};

//! The constraint to enforce a minimum value.
template <std::totally_ordered T>
class Min {
public:
    static constexpr OptType type {OptType::Min};

    explicit constexpr Min(T val,
                           const BoundType bound = BoundType::Closed) noexcept :
        val_ {std::move(val)}, bound_ {bound} {}

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (const auto invalid {bound_ == BoundType::Closed ? val < val_
                                                                : val <= val_};
                invalid) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::result_out_of_range),
                     BuildErrorMsg(val_)}};
            } else {
                return val;
            }
        });
    }

private:
    std::string BuildErrorMsg(const T& val) const noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format(
                "The value {} is less than the minimum allowed value {}", val,
                val_);
        } else {
            return "The value is less than the minimum allowed value";
        }
    }

    BoundType bound_;
    T val_;
};

//! The constraint to enforce a maximum value.
template <std::totally_ordered T>
class Max {
public:
    static constexpr OptType type {OptType::Max};

    explicit constexpr Max(T val,
                           const BoundType bound = BoundType::Closed) noexcept :
        val_ {std::move(val)}, bound_ {bound} {}

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (const auto invalid {bound_ == BoundType::Closed ? val > val_
                                                                : val >= val_};
                invalid) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::result_out_of_range),
                     BuildErrorMsg(val_)}};
            } else {
                return val;
            }
        });
    }

private:
    std::string BuildErrorMsg(const T& val) const noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format(
                "The value {} is greater than the maximum allowed value {}",
                val, val_);
        } else {
            return "The value is greater than the maximum allowed value";
        }
    }

    BoundType bound_;
    T val_;
};

//! The constraint to check if a value satisfies a predicate.
template <typename T>
class Predicate {
public:
    using FuncType = std::function<bool(const T&)>;

    static constexpr OptType type {OptType::Predicate};

    explicit Predicate(FuncType func) noexcept : func_ {std::move(func)} {
        assert(func_ != nullptr);
    }

    //! Apply the predicate function to a given value.
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (!func_(val)) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::invalid_argument),
                     BuildErrorMsg(val)}};
            } else {
                return val;
            }
        });
    }

private:
    static std::string BuildErrorMsg(const T& val) noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format("The value {} does not satisfy the predicate",
                               val);
        } else {
            return "The value does not satisfy the predicate";
        }
    }

    FuncType func_;
};

//! The constraint to clamp a value to a specified range.
template <std::totally_ordered T>
class Clamp {
public:
    static constexpr OptType type {OptType::Clamp};

    explicit constexpr Clamp(std::pair<T, T> range) noexcept :
        range_ {std::move(range)} {
        assert(range_.first <= range_.second);
    }

    //! Same as @p std::clamp.
    constexpr ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            return std::clamp(val, range_.first, range_.second);
        });
    }

private:
    std::pair<T, T> range_;
};

//! The constraint to check if a value is within a specified range.
template <std::totally_ordered T>
class InRange {
public:
    static constexpr OptType type {OptType::InRange};

    explicit constexpr InRange(std::pair<T, T> range,
                               const std::pair<BoundType, BoundType> bounds =
                                   {BoundType::Closed,
                                    BoundType::Closed}) noexcept :
        range_ {std::move(range)},
        bounds_ {bounds},
        brackets_ {BuildBoundBrackets(bounds)} {
        assert(range_.first <= range_.second);
    }

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            const auto& [min, max] {range_};
            const auto [left_bound, right_bound] {bounds_};
            const auto less_than_min {
                (left_bound == BoundType::Closed) ? val < min : val <= min};
            const auto greater_than_max {
                (right_bound == BoundType::Closed) ? val > max : val >= max};
            if (less_than_min || greater_than_max) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::result_out_of_range),
                     BuildErrorMsg(val)}};
            } else {
                return val;
            }
        });
    }

private:
    std::string BuildErrorMsg(const T& val) const noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format(
                "The value {} is out of the allowed range {}{}, {}{}", val,
                brackets_.first, range_.first, range_.second, brackets_.second);
        } else {
            return "The value is out of the allowed range";
        }
    }

    std::pair<char, char> brackets_;
    std::pair<BoundType, BoundType> bounds_;
    std::pair<T, T> range_;
};

//! The constraint to check if a value is not within a specified range.
template <std::totally_ordered T>
class NotInRange {
public:
    static constexpr OptType type {OptType::NotInRange};

    explicit constexpr NotInRange(std::pair<T, T> range,
                                  const std::pair<BoundType, BoundType> bounds =
                                      {BoundType::Closed,
                                       BoundType::Closed}) noexcept :
        range_ {std::move(range)},
        bounds_ {bounds},
        brackets_ {BuildBoundBrackets(bounds)} {
        assert(range_.first <= range_.second);
    }

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            const auto& [min, max] {range_};
            const auto [left_bound, right_bound] {bounds_};
            if (const auto in_range {
                    (left_bound == BoundType::Closed ? val >= min : val > min)
                    && (right_bound == BoundType::Closed ? val <= max
                                                         : val < max)};
                in_range) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::result_out_of_range),
                     BuildErrorMsg(val)}};
            } else {
                return val;
            }
        });
    }

private:
    std::string BuildErrorMsg(const T& val) const noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format(
                "The value {} is within the forbidden range {}{}, {}{}", val,
                brackets_.first, range_.first, range_.second, brackets_.second);
        } else {
            return "The value is within the forbidden range";
        }
    }

    std::pair<char, char> brackets_;
    std::pair<BoundType, BoundType> bounds_;
    std::pair<T, T> range_;
};

//! Whether a type can be used as a key in @p std::unordered_set.
template <typename T>
concept IsHashable = requires(const T& v) {
    { std::hash<T> {}(v) };
} && std::equality_comparable<T>;

//! The constraint to check if a value is in a specified set.
template <IsHashable T>
class InSet {
public:
    static constexpr OptType type {OptType::InSet};

    explicit InSet(std::initializer_list<T> vals) noexcept :
        vals_ {std::move(vals)} {}

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (!vals_.contains(val)) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::result_out_of_range),
                     BuildErrorMsg(val)}};
            } else {
                return val;
            }
        });
    }

private:
    static std::string BuildErrorMsg(const T& val) noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format("The value {} is not in the allowed set", val);
        } else {
            return "The value is not in the allowed set";
        }
    }

    std::unordered_set<T> vals_;
};

//! The constraint to check if a value is not in a specified set.
template <IsHashable T>
class NotInSet {
public:
    static constexpr OptType type {OptType::NotInSet};

    explicit NotInSet(std::initializer_list<T> vals) noexcept :
        vals_ {std::move(vals)} {}

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (vals_.contains(val)) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::result_out_of_range),
                     BuildErrorMsg(val)}};
            } else {
                return val;
            }
        });
    }

private:
    static std::string BuildErrorMsg(const T& val) noexcept {
        if constexpr (FormatInvocable<T>) {
            return std::format("The value {} is in the prohibited set", val);
        } else {
            return "The value is in the prohibited set";
        }
    }

    std::unordered_set<T> vals_;
};

//! The constraint to check if a value is not null or @p false.
template <std::convertible_to<bool> T>
class NotNull {
public:
    static constexpr OptType type {OptType::NotNull};

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (!static_cast<bool>(val)) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::invalid_argument),
                     "The value cannot be null"}};
            } else {
                return val;
            }
        });
    }
};

//! The constraint to check if a container is not empty.
template <typename T>
    requires requires(const T& v) {
        { v.empty() } -> std::convertible_to<bool>;
    }
class NotEmpty {
public:
    static constexpr OptType type {OptType::NotEmpty};

    //! @copydoc Enum::Apply
    ChainType<T> Apply(const ChainType<T>& var) const noexcept {
        return var.and_then([this](const T& val) noexcept -> ChainType<T> {
            if (val.empty()) [[unlikely]] {
                return std::unexpected<Error> {
                    {std::make_error_code(std::errc::invalid_argument),
                     "The value cannot be empty"}};
            } else {
                return val;
            }
        });
    }
};

//! The constraint to perform custom transformations.
template <typename From, typename To>
class Transformer {
public:
    using FuncType = std::function<ChainType<To>(const From&)>;

    static constexpr OptType type {OptType::Transformer};

    explicit Transformer(FuncType func) noexcept : func_ {std::move(func)} {
        assert(func_ != nullptr);
    }

    /**
     * @brief Apply the transformation function to a given value.
     *
     * @note
     * In most cases, the parameter type of @p Apply is exactly the same as the return type of the previous constraint in the chain.
     * For example, the return type of @p Set<int>::Apply and the parameter type of `Transformer<int, bool>::Apply` are both @p ChainType<int>.
     * In this case, if the previous constraint returns an @p std::unexpected, the user-provided transformation function will not be called.
     * Instead, the same @p std::unexpected object will be returned directly.
     *
     * However, in some cases, we may want the transformation function to return the validity of the previous constraint's return value.
     * For example, we might need to set a boolean value to @p true if a number exists in a given set.
     * The constraint chain of @p Set<int> and `Transformer<int, bool>` does not work because the when the number is not in the set, the transformer will be skipped.
     * We should use `Transformer<ChainType<int>, bool>` and the following function:
     *
     * @code {.cpp}
     * constexpr bool IsValid(const ChainType<int>& val) noexcept {
     *     return val.has_value();
     * }
     * @endcode
     *
     * The parameter type of `Transformer<ChainType<int>, bool>::Apply` is @p ChainType<ChainType<int>>.
     * Regardless of whether @p Set<int> returns a number or an @p std::unexpected, the result will always be forwarded to the user-provided function.
     */
    ChainType<To> Apply(const ChainType<From>& var) const noexcept {
        return var.and_then([this](const From& val) noexcept -> ChainType<To> {
            return func_(val);
        });
    }

private:
    FuncType func_;
};

//! The namespace for constraint checks.
namespace chk {

//! Whether the given type is a valid constraint type.
template <typename T>
concept IsOpt = requires {
    { T::type } -> std::convertible_to<OptType>;
};

//! Extract the constraint type from the given constraint class.
template <IsOpt T>
inline constexpr OptType opt_type {T::type};

//! Whether the given type is a valid constraint type and has the specified @ref Type.
template <typename T, OptType type>
concept IsSameOpt = IsOpt<T> && (opt_type<T> == type);

//! Whether the given type is the specified constraint type.
#define DEFINE_OPT_TYPE_CHECKER(name, type) \
    template <IsOpt T>                      \
    inline constexpr bool is_##name = IsSameOpt<T, type>;

DEFINE_OPT_TYPE_CHECKER(min, OptType::Min)
DEFINE_OPT_TYPE_CHECKER(max, OptType::Max)
DEFINE_OPT_TYPE_CHECKER(clamp, OptType::Clamp)
DEFINE_OPT_TYPE_CHECKER(in_range, OptType::InRange)
DEFINE_OPT_TYPE_CHECKER(not_in_range, OptType::NotInRange)
DEFINE_OPT_TYPE_CHECKER(not_null, OptType::NotNull)
DEFINE_OPT_TYPE_CHECKER(not_empty, OptType::NotEmpty)
DEFINE_OPT_TYPE_CHECKER(transformer, OptType::Transformer)
DEFINE_OPT_TYPE_CHECKER(enum, OptType::Enum)
DEFINE_OPT_TYPE_CHECKER(predicate, OptType::Predicate)
DEFINE_OPT_TYPE_CHECKER(in_set, OptType::InSet)
DEFINE_OPT_TYPE_CHECKER(not_in_set, OptType::NotInSet)

//! Whether a given constraint type defines an @p Apply method.
template <typename Opt, typename T>
concept HasApplyMethod = requires(const Opt& opt, const T& val) {
    { opt.Apply(val) };
};

}  // namespace chk

}  // namespace opt

/**
 * @brief The chain of constraints applied to a value.
 *
 * @tparam Out The final output type after applying all constraints.
 * @tparam Opts Constraint types.
 */
template <typename Out, opt::chk::IsOpt... Opts>
class ConstraintChain {
public:
    using Type = Out;

    //! Constructs a chain with the provided constraints.
    explicit constexpr ConstraintChain(Opts... opts) noexcept :
        opts_ {std::move(opts)...} {}

    //! Apply all constraints to a given value in order.
    constexpr ChainType<Type> Apply(const auto& val) const noexcept {
        return ApplyAll<0>(val);
    }

    constexpr void swap(ConstraintChain& o) noexcept {
        std::ranges::swap(opts_, o.opts_);
    }

private:
    template <std::size_t i = 0, typename U>
    constexpr ChainType<Type> ApplyAll(const U& val) const noexcept {
        if constexpr (i >= sizeof...(Opts)) {
            return val;
        } else {
            const auto& opt {std::get<i>(opts_)};
            using OptType = std::decay_t<decltype(opt)>;
            if constexpr (opt::chk::HasApplyMethod<OptType, U>) {
                return ApplyAll<i + 1>(opt.Apply(val));
            } else {
                return ApplyAll<i + 1>(val);
            }
        }
    }

    std::tuple<std::decay_t<Opts>...> opts_;
};

/**
 * @brief The chain of constraints applied to a value and check its validity finally.
 *
 * @tparam Out The final output type to validate after applying all constraints.
 * @tparam Opts Constraint types.
 */
template <typename Out, opt::chk::IsOpt... Opts>
class ValidationChain {
public:
    using Type = bool;

    //! Constructs a chain with the provided constraints.
    explicit constexpr ValidationChain(Opts... opts) noexcept :
        opts_ {std::move(opts)...} {}

    //! Apply all constraints to a given value in order and return its validity.
    constexpr bool Apply(const auto& val) const noexcept {
        return opts_.Apply(val).has_value();
    }

    constexpr void swap(ValidationChain& o) noexcept {
        opts_.swap(o.opts_);
    }

private:
    ConstraintChain<Out, Opts...> opts_;
};

//! The constrained variable that applies one or more constraints when setting a value.
template <typename T, opt::chk::IsOpt... Opts>
class ConstrainedVariable {
public:
    using Type = T;

    //! Constructs a variable with the provided constraints.
    explicit constexpr ConstrainedVariable(Opts... opts) noexcept :
        opts_ {std::move(opts)...} {}

    //! Constructs a variable with the provided initial value and constraints.
    explicit constexpr ConstrainedVariable(Type val, Opts... opts) :
        ConstrainedVariable {std::move(opts)...} {
        if (auto new_val {Set(std::move(val))}; !new_val.has_value()) {
            throw std::runtime_error {std::move(new_val).error().second};
        }
    }

    //! Get the current value.
    constexpr Type Get() const noexcept {
        return val_;
    }

    /**
     * @brief Attempt to set a new value after applying all applicable constraints.
     *
     * @param val The input value to be validated and possibly transformed.
     * @return The final valid value or an error.
     */
    template <typename U = Type>
    constexpr ChainType<Type> Set(const U& val) noexcept {
        return opts_.Apply(val).transform([this](auto&& final) noexcept {
            val_ = static_cast<Type>(std::forward<decltype(final)>(final));
            return val_;
        });
    }

    constexpr void swap(ConstrainedVariable& o) noexcept {
        std::ranges::swap(val_, o.val_);
        opts_.swap(o.opts_);
    }

private:
    ConstraintChain<T, Opts...> opts_;
    Type val_ {};
};

//! The boolean variable validated against one or more constraints.
template <typename T, opt::chk::IsOpt... Opts>
class ValidBoolVariable {
    static constexpr bool IsValid(const ChainType<T>& val) noexcept {
        return val.has_value();
    }

public:
    using Type = bool;

    //! Constructs a variable with the provided constraints.
    explicit constexpr ValidBoolVariable(Opts... opts) noexcept :
        val_ {std::move(opts)...,
              opt::Transformer<ChainType<T>, bool> {IsValid}} {}

    //! Get the current validity.
    constexpr bool Get() const noexcept {
        return val_.Get();
    }

    //! Set the validity of a new value after applying all constraints.
    constexpr bool Set(const T& val) noexcept {
        const auto ret {val_.Set(val)};
        assert(ret.has_value());
        return ret.value();
    }

    constexpr void swap(ValidBoolVariable& o) noexcept {
        val_.swap(o.val_);
    }

private:
    ConstrainedVariable<bool, Opts..., opt::Transformer<ChainType<T>, bool>>
        val_;
};

}  // namespace cv