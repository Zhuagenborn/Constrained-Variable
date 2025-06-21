# Constrained Variable

![C++](docs/badges/C++.svg)
[![CMake](docs/badges/Made-with-CMake.svg)](https://cmake.org)
![GitHub Actions](docs/badges/Made-with-GitHub-Actions.svg)
![License](docs/badges/License-MIT.svg)

## Introduction

A header-only library written in *C++23* for applying composable validation and transformation constraints to variables in a flexible and type-safe way, supporting:

- Range checks.
- Null and emptiness checks.
- Enumeration restrictions.
- Value transformation.
- Custom predicate validation.

Constraints are applied in a user-defined order and ensure the values are within the expected bounds or rules.
Helpful error messages are generated when constraints are violated.

## Unit Tests

### Prerequisites

- Install *GoogleTest*.
- Install *CMake*.

### Building

Go to the project folder and run:

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### Running

Go to the `build` folder and run:

```bash
ctest -VV
```

## Examples

See more examples in `tests/constrained_var_tests.cpp`.

```c++
// Create a constraint.
constexpr InRange<std::size_t> speed_range {{10, 80}};

// Create a constrained variable.
InRangeVariable<std::size_t> speed {speed_range};

// Try to set a new value.
speed.Set(input).transform_error([](const Error& err) noexcept {
    std::println("Failed to set the speed: {}", err.second);
    return false;
});
```

### Constraints

#### `Min`

`Min` ensures the value is not less than a specified minimum.

```c++
constexpr Min<int> min_opt {10};
MinVariable<int> var {min_opt};

EXPECT_FALSE(var.Set(5).has_value());
EXPECT_EQ(var.Set(15).value_or(0), 15);
```

#### `Max`

`Max` ensures the value does not exceed a specified maximum.

```c++
constexpr Max<int> max_opt {100};
MaxVariable<int> var {max_opt};

EXPECT_FALSE(var.Set(120).has_value());
EXPECT_EQ(var.Set(80).value_or(0), 80);
```

#### `InRange`

`InRange` ensures the value lies within a specified range.

```c++
constexpr InRange<int> in_range_opt {{10, 100}};
InRangeVariable<int> var {in_range_opt};

EXPECT_FALSE(var.Set(5).has_value());
EXPECT_EQ(var.Set(50).value_or(0), 50);
```

#### `NotInRange`

`NotInRange` ensures the value is not within a specified range.

```c++
constexpr NotInRange<int> not_in_range_opt {{10, 100}};
NotInRangeVariable<int> var {not_in_range_opt};

EXPECT_FALSE(var.Set(50).has_value());
EXPECT_EQ(var.Set(5).value_or(0), 5);
```

#### `InSet`

`InSet` ensures the value is in a specified set.

```c++
const InSet<int> in_set_opt {1, 2};
InSetVariable<int> var {in_set_opt};

EXPECT_EQ(var.Set(1).value_or(0), 1);
EXPECT_EQ(var.Set(2).value_or(0), 2);
EXPECT_FALSE(var.Set(3).has_value());
```

#### `NotInSet`

`NotInSet` ensures the value is not in a specified set.

```c++
const NotInSet<int> not_in_set_opt {1, 2};
NotInSetVariable<int> var {not_in_set_opt};

EXPECT_FALSE(var.Set(1).has_value());
EXPECT_FALSE(var.Set(2).has_value());
EXPECT_EQ(var.Set(3).value_or(0), 3);
```

#### `Clamp`

`Clamp` clamps the value into a specified range (`std::clamp`).

```c++
constexpr Clamp<int> clamp_opt {{10, 100}};
ClampVariable<int> var {clamp_opt};

EXPECT_EQ(var.Set(5).value_or(0), 10);
EXPECT_EQ(var.Set(120).value_or(0), 100);
```

#### `Enum`

`Enum` ensures the enumeration lies within an inclusive range.

```c++
enum class Color { White, Red, Green, Black, Invalid };

template <>
struct EnumValues<Color> {
    static constexpr std::array values {Color::White, Color::Red, Color::Green, Color::Black};
};

EnumVariable<Color> var;

EXPECT_FALSE(var.Set(Color::Invalid).has_value());
EXPECT_EQ(var.Set(Color::Red).value_or(Color::Invalid), Color::Red);
```

#### `NotNull`

`NotNull` ensures the value is not null or `false`.

```c++
NotNullVariable<int> var;

EXPECT_FALSE(var.Set(0).has_value());
EXPECT_EQ(var.Set(42).value_or(0), 42);
```

#### `NotEmpty`

`NotEmpty` ensures the container like `std::vector` is not empty.

```c++
NotEmptyVariable<std::vector<int>> var;

EXPECT_FALSE(var.Set(std::vector<int> {}).has_value());
EXPECT_EQ(var.Set(std::vector<int> {1}).value_or(std::vector<int> {}), std::vector<int> {1});
```

#### `Predicate`

`Predicate` ensures the value satisfies a predicate.

```c++
const Predicate<int> pred {[](const int x) noexcept {
    return x % 2 == 0;
}};

PredicateVariable<int> var {pred};

EXPECT_FALSE(var.Set(3).has_value());
EXPECT_EQ(var.Set(4).value_or(0), 4);
```

#### `Transformer`

`Transformer` transforms the value before validation or storage, supporting chaining with other constraints.

```c++
const Transformer<int, double> func {[](const int v) noexcept {
    return static_cast<double>(v);
}};

TransformerVariable<int, double> var {func};

EXPECT_EQ(var.Set(5).value_or(0.0), 5.0);
```

### Constraint Chains

This example transforms an integer to a vector as its size and then check if it is empty using a chain of `Transformer` and `NotEmpty`.

```c++
ChainType<std::vector<int>> SizeToVector(const std::size_t& size) noexcept {
    return std::vector<int>(size);
}

constexpr NotEmpty<std::vector<int>> not_empty_opt;
const Transformer<std::size_t, std::vector<int>> transformer_opt {SizeToVector};
ConstrainedVariable<std::vector<int>, decltype(transformer_opt), decltype(not_empty_opt)> var {transformer_opt, not_empty_opt};

EXPECT_FALSE(var.Set(0).has_value());
EXPECT_EQ(var.Set(1).value_or(std::vector<int> {}), std::vector<int>(1));
```

This example defines a boolean variable that is `true` only when the input value is `1` or `2` using a chain of `InSet` and `Transformer`.

```c++
const InSet<int> in_set_opt {1, 2};
const Transformer<ChainType<int>, bool> transformer_opt {
    [](const ChainType<int>& val) noexcept {
        return val.has_value();
    }
};

ConstrainedVariable<bool, decltype(in_set_opt), decltype(transformer_opt)> var {in_set_opt, transformer_opt};

EXPECT_TRUE(var.Set(1).has_value());
EXPECT_TRUE(var.Get());

EXPECT_TRUE(var.Set(3).has_value());
EXPECT_FALSE(var.Get());
```

In most cases, the parameter type of `Apply` is exactly the same as the return type of the previous constraint in the chain.
For example, the return type of `Set<int>::Apply` and the parameter type of `Transformer<int, bool>::Apply` are both `ChainType<int>`.
In this case, if the previous constraint returns an `std::unexpected`, the user-provided transformation function will not be called.

But currently we want the transformation function to return the validity of the previous constraint's return value.
The constraint chain of `Set<int>` and `Transformer<int, bool>` does not work because the when the number is not in the set, the transformer will be skipped.
Instead, we should use `Transformer<ChainType<int>, bool>`.
The parameter type of its `Apply` is `ChainType<ChainType<int>>`.
Regardless of whether `Set<int>` returns a number or an `std::unexpected`, the result will always be forwarded to the user-provided function.

### Validated Boolean Variables

You can directly use `ValidatedBoolVariable` if you need a boolean variable validated against a set of constraints.

```c++
const InSet<int> in_set_opt {1, 2};
ValidatedBoolVariable<int, decltype(in_set_opt)> var {in_set_opt};

EXPECT_TRUE(var.Set(1));
EXPECT_TRUE(var.Set(2));
EXPECT_FALSE(var.Set(3));
```

### Validation

If you only need to validate values without storing them, you can directly `ConstraintChain` and `ValidationChain`.

```c++
const InSet<int> in_set_opt {1, 2};
ValidationChain<int, decltype(in_set_opt)> var {in_set_opt};

EXPECT_TRUE(var.Apply(1));
EXPECT_TRUE(var.Apply(2));
EXPECT_FALSE(var.Apply(3));
```

## License

Distributed under the *MIT License*. See `LICENSE` for more information.