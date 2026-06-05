// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STRING_MANIPULATION_HPP
#define KOKKOS_STRING_MANIPULATION_HPP

#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Assert.hpp>

namespace Kokkos {
namespace Impl {

// This header provides a subset of the functionality from <cstring>.  In
// contrast to the standard library header, functions are usable on the device
// and in constant expressions.  It also includes functionality from <charconv>
// to convert an integer value to a character sequence.

//<editor-fold desc="String examination">
// returns the length of a given string
KOKKOS_INLINE_FUNCTION constexpr std::size_t strlen(const char *str) {
  std::size_t i = 0;
  while (str[i] != '\0') {
    ++i;
  }
  return i;
}

// compares two strings
KOKKOS_INLINE_FUNCTION constexpr int strcmp(const char *lhs, const char *rhs) {
  while (*lhs == *rhs++) {
    if (*lhs++ == '\0') {
      return 0;
    }
  }
  return static_cast<unsigned int>(*lhs) -
         static_cast<unsigned int>(*(rhs - 1));
}

// compares a certain number of characters from two strings
KOKKOS_INLINE_FUNCTION constexpr int strncmp(const char *lhs, const char *rhs,
                                             std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (lhs[i] != rhs[i]) {
      return lhs[i] < rhs[i] ? -1 : 1;
    } else if (lhs[i] == '\0') {
      return 0;
    }
  }
  return 0;
}
//</editor-fold>

//<editor-fold desc="String manipulation">
// copies one string to another
KOKKOS_INLINE_FUNCTION constexpr char *strcpy(char *dest, const char *src) {
  char *d = dest;
  for (; (*d = *src) != '\0'; ++d, ++src) {
  }
  return dest;
}

// copies a certain amount of characters from one string to another
KOKKOS_INLINE_FUNCTION constexpr char *strncpy(char *dest, const char *src,
                                               std::size_t count) {
  if (count != 0) {
    char *d = dest;
    do {
      if (char const c = (*d++ = *src++); c == '\0') {
        while (--count != 0) {
          *d++ = '\0';
        }
        break;
      }
    } while (--count != 0);
  }
  return dest;
}

// concatenates two strings
KOKKOS_INLINE_FUNCTION constexpr char *strcat(char *dest, const char *src) {
  char *d = dest;
  for (; *d != '\0'; ++d) {
  }
  while ((*d++ = *src++) != '\0') {
  }
  return dest;
}

// concatenates a certain amount of characters of two strings
KOKKOS_INLINE_FUNCTION constexpr char *strncat(char *dest, const char *src,
                                               std::size_t count) {
  if (count != 0) {
    char *d = dest;
    for (; *d != '\0'; ++d) {
    }
    do {
      if (char const c = (*d = *src++); c == '\0') {
        break;
      }
      d++;
    } while (--count != 0);
    *d = '\0';
  }
  return dest;
}
//</editor-fold>

//<editor-fold desc="Character conversions">
template <std::unsigned_integral Unsigned>
KOKKOS_FUNCTION constexpr unsigned int to_chars_len(Unsigned val) {
  unsigned int const base = 10;
  unsigned int n          = 1;
  while (val >= base) {
    val /= base;
    ++n;
  }
  return n;
}

template <std::floating_point FloatType>
KOKKOS_FUNCTION unsigned int to_chars_len(FloatType f) {
  using uint_t            = Kokkos::equivalent_int_t<FloatType>;
  constexpr int mantissa_bits = Kokkos::mantissa_bits_v<FloatType>;
  constexpr int exponent_bits = Kokkos::exponent_bits_v<FloatType>;

  constexpr uint_t exp_mask      = (uint_t(1) << exponent_bits) - 1;
  constexpr uint_t mantissa_mask = (uint_t(1) << mantissa_bits) - 1;

  uint_t u = Kokkos::bit_cast<uint_t>(f);

  // Extract double infos
  uint_t sign = u & (uint_t(1) << (mantissa_bits + exponent_bits));
  uint_t exp  = ((u & (exp_mask << mantissa_bits))) >> mantissa_bits;

  unsigned int ret = 0;

  ret += sign ? 1 : 0;
  if (exp == exp_mask) {
    // 'inf' and 'nan' are both 3 chars
    return ret + 3;
  }

  // Only a single format is supported:
  //  - 1 significant number
  //  - 1 decimal point ('.')
  //  - 6 fractional numbers
  //  - 1 exponent marker ('e')
  //  - 1 exponent sign ('+' or '-')
  //  - 2 digits padded with 0 for the exponent if it is between -99 and 99, 3
  //  digits if it is bigger

  if constexpr (std::is_same_v<FloatType, double>) {
    // Check whether the exponent has two or three decimal digits by comparing
    // with the double that is the tipping point:
    //  - 0x2b617f7d402b1834 is the first number that rounds to 1.0000000e-99,
    //  previous number 0x2b617f7d402b1833 rounds to 9.9999999e-100
    //  - 0x54b249ad163d7d25 is the last number that rounds to 9.9999999e+99,
    //  0x54b249ad163d7d26 rounds to 1.0000000e+100
    u = u & (exp_mask << mantissa_bits | mantissa_mask);
    if ((u != 0 && u < 0x2b617f7d402b1834) || u > 0x54b249ad163d7d25) {
      return ret + 13;
    }
  }
  return ret + 12;
}

template <class Unsigned>
KOKKOS_FUNCTION constexpr void to_chars_impl(char *first, unsigned int len,
                                             Unsigned val) {
  unsigned int const base = 10;
  static_assert(std::is_integral_v<Unsigned>, "implementation bug");
  static_assert(std::is_unsigned_v<Unsigned>, "implementation bug");
  unsigned int pos = len - 1;
  while (val > 0) {
    auto const num = val % base;
    val /= base;
    first[pos] = '0' + num;
    --pos;
  }
}

// define values of portable error conditions that correspond to the POSIX error
// codes
enum class errc {
  ok              = 0,
  value_too_large = 75  // equivalent POSIX error is EOVERFLOW
};
struct to_chars_result {
  char *ptr;
  errc ec;
};

// converts an integer value to a character sequence
template <class Integral>
KOKKOS_FUNCTION constexpr to_chars_result to_chars_i(char *first, char *last,
                                                     Integral value) {
  // NOLINTBEGIN(bugprone-invalid-enum-default-initialization)
  using Unsigned = std::conditional_t<sizeof(Integral) <= sizeof(unsigned int),
                                      unsigned int, unsigned long long>;
  Unsigned unsigned_val = value;  // NOLINT(bugprone-signed-char-misuse)
  if (value == 0) {
    *first = '0';
    return {first + 1, {}};
  } else if constexpr (std::is_signed_v<Integral>) {
    if (value < 0) {
      *first++     = '-';
      unsigned_val = Unsigned(~value) + Unsigned(1);
    }
  }
  std::ptrdiff_t const len = to_chars_len(unsigned_val);
  if (last - first < len) {
    return {last, errc::value_too_large};
  }
  to_chars_impl(first, len, unsigned_val);
  return {first + len, {}};
  // NOLINTEND(bugprone-invalid-enum-default-initialization)
}
//</editor-fold>

/*
 * Decimal representation of floating point numbers
 */
template <typename FloatType, std::size_t size>
struct DecimalRepresentation {
  using uint_t = Kokkos::equivalent_int_t<FloatType>;

  // Buffer that contains the decimal representation of a number in scientific
  // notation
  uint8_t buffer[size];
  // Exponent of the decimal representation stored
  int exp10;
  // For instance, if the number stored is 2^10, `buffer` contains
  // "102400000..." and `exp10` is 3

  KOKKOS_FUNCTION constexpr DecimalRepresentation() : buffer{0}, exp10{0} {}

  // This can't be higher than 60 since we need to be able to store up to
  // (2^max_div - 1) * 9 in the remainder, bigger number wouldn't fit in a
  // uint64_t
  static constexpr int max_div = 60;
  static constexpr int max_mul = 60;

  // Print buffer for debugging purpose
  void print() {
    char tmp[size + 1];
    for (int i = 0; i < size; ++i) {
      tmp[i] = buffer[i] + '0';
    }
    tmp[size] = '\0';
    Kokkos::printf("%se%i", tmp, exp10);
  }

  // Round number up (equivalent to adding 10 ^ (exp10 - size))
  KOKKOS_FUNCTION void round_up(size_t index = size - 1) {
    bool carry = true;
    int i     = index;

    while (i >= 0 && carry) {
      if (++buffer[i] > 9) {
        buffer[i] = 0;
      } else {
        carry = false;
      }
      --i;
    }

    // Initial number had the form "999....999", rounds to "100....000" with
    // `exp10`+1
    if (carry) {
      shift_right(1);
    }
  }

  // Shift the decimal representation in buffer shift time to the right,
  // inserting `insert` as the leading numbers. Returns the decimal
  // representation of the most significant number that was shifted out
  KOKKOS_FUNCTION uint8_t shift_right(uint8_t insert, int shift = 1) {
    if (shift < 1) {
      // nothing to do
      return 0;
    }

    uint8_t shifted_out = buffer[size - shift];

    int j = size - 1;
    while (j >= shift) {
      buffer[j] = buffer[j - shift];
      --j;
    }
    while (j >= 0) {
      buffer[j] = insert;
      --j;
    }

    exp10 += shift;
    return shifted_out;
  }

  // Divide decimal number by 2^exp
  KOKKOS_FUNCTION void divide_by_power_of_two(int exp) {
    KOKKOS_ASSERT(exp <= max_div);

    uint64_t dividend   = 0x0llu;
    uint64_t divisor    = 0x1llu << exp;
    std::size_t src_idx = 0;
    std::size_t res_idx = 0;

    bool first_pass = true;

    do {
      do {
        dividend *= 10;
        if (src_idx < size) {
          dividend += buffer[src_idx++];
        }
        if (dividend != 0 && dividend / divisor != 0) {
          break;
        } else {
          if (first_pass) {
            --exp10;
          }
          if (dividend != 0 && !first_pass) {
            buffer[res_idx++] = 0;
          }
        }
      } while (dividend != 0 && res_idx < size);

      first_pass = false;

      if (res_idx < size) {
        buffer[res_idx++] = dividend / divisor;
        dividend %= divisor;
      }
    } while (res_idx < size);

    if (dividend * 2 > divisor) {
      round_up();
    }
  }

  // Multiply decimal number by 2^exp
  KOKKOS_FUNCTION void multiply_by_power_of_two(int exp) {
    KOKKOS_ASSERT(exp <= max_mul);

    uint64_t multiplior = 0x1llu << exp;
    uint64_t carry      = 0llu;

    for (int i = size - 1; i >= 0; --i) {
      uint64_t tmp = buffer[i] * multiplior + carry;
      buffer[i]    = tmp % 10;
      carry        = tmp / 10;
    }

    uint8_t discarded = 0;
    // Continue adding the carry to the computed number
    while (carry > 0) {
      discarded = shift_right(carry % 10);
      carry /= 10;
    }

    // Round if the last digit shifted out was >= 5
    if (discarded >= 5) {
      round_up();
    }
  }

  template <size_t size_r>
  KOKKOS_FUNCTION DecimalRepresentation<FloatType, size> &operator+=(
      DecimalRepresentation<FloatType, size_r> rhs) {
    KOKKOS_ASSERT(rhs.exp10 == exp10);

    int carry = 0;
    for (int i = size - 1; i >= 0; --i) {
      buffer[i] += rhs.buffer[i] + carry;
      if (buffer[i] > 9) {
        carry = 1;
        buffer[i] -= 10;
      } else {
        carry = 0;
      }
    }

    if (carry) {
      if (shift_right(1) >= 5) {
        round_up();
      }
    }

    return *this;
  }
};

/**
 * Decimal representation of a power of two
 */
template <class FloatType, std::size_t size>
struct BaseTwoExponent : public DecimalRepresentation<FloatType, size> {
  using DecimalRepresentation<FloatType, size>::buffer;
  using DecimalRepresentation<FloatType, size>::exp10;
  using DecimalRepresentation<FloatType, size>::max_div;
  using DecimalRepresentation<FloatType, size>::max_mul;

  static constexpr int bias =
      (typename Kokkos::equivalent_int_t<FloatType>(1)
       << (Kokkos::exponent_bits<FloatType>::value - 1)) +
      Kokkos::mantissa_bits<FloatType>::value - 2;

  // Exponent of the power stored in `buffer`
  // Stored with the total bias added
  int exp2;

  // Initialize with the decimal representation of 2^0 = 1
  KOKKOS_FUNCTION constexpr BaseTwoExponent() {
    DecimalRepresentation<FloatType, size>::buffer[0] = 1;
    exp2                                              = bias;
  }

  void print() {
    DecimalRepresentation<FloatType, size>::print();
    Kokkos::printf(" = 2^%i", exp2 - bias);
  }

  KOKKOS_FUNCTION void generate_nth_exp(int exp2_target) {
    if (exp2_target < exp2) {
      // Divide max_div by max_div until we can reach the target with one
      // smaller step
      while (exp2 - max_div > exp2_target) {
        DecimalRepresentation<FloatType, size>::divide_by_power_of_two(max_div);
        exp2 -= max_div;
      }

      // Last division needed to reach the target
      if (exp2 != exp2_target) {
        DecimalRepresentation<FloatType, size>::divide_by_power_of_two(
            exp2 - exp2_target);
        exp2 -= (exp2 - exp2_target);
      }
    } else if (exp2_target > exp2) {
      // Multiply max_mul by max_mul until we can reach the target with one
      // smaller step
      while (exp2 + max_mul < exp2_target) {
        DecimalRepresentation<FloatType, size>::multiply_by_power_of_two(
            max_mul);
        exp2 += max_mul;
      }

      // Last multiplication needed to reach the target
      if (exp2 != exp2_target) {
        DecimalRepresentation<FloatType, size>::multiply_by_power_of_two(
            exp2_target - exp2);
      }
    }
    exp2 = exp2_target;
  }
};

template <typename FloatType>
KOKKOS_FUNCTION to_chars_result to_chars_f(char *first, char *last,
                                           FloatType f) {
  static_assert(std::is_same_v<FloatType, double> ||
                std::is_same_v<FloatType, float>);

  using uint_t            = Kokkos::equivalent_int_t<FloatType>;
  constexpr int mantissa_bits = Kokkos::mantissa_bits_v<FloatType>;
  constexpr int exponent_bits = Kokkos::exponent_bits_v<FloatType>;

  constexpr uint_t exp_mask      = (uint_t(1) << exponent_bits) - 1;
  constexpr uint_t mantissa_mask = (uint_t(1) << mantissa_bits) - 1;

  // Number of decimal digits used for internal computation
  // These numbers don't depend on the input type, but on the number of output
  // digits.
  // They were found empirically so that `to_chars_f` has the same output as
  // std::printf("%e")
  constexpr int precision     = 18;
  constexpr int exp_precision = 22;

  std::ptrdiff_t const len = to_chars_len(f);
  if (last - first < len) {
    return {last, errc::value_too_large};
  }
  uint_t u = Kokkos::bit_cast<uint_t>(f);

  // Extract double informations
  uint_t sign     = u & (uint_t(1) << (mantissa_bits + exponent_bits));
  uint_t mantissa = u & mantissa_mask;
  uint_t exp      = (u >> mantissa_bits) & exp_mask;

  char *out = first;

  // Add sign to output if needed
  if (sign) {
    *out++ = '-';
  }

  // Inf and NaN
  if (exp == exp_mask) {
    if (mantissa) {
      strcpy(out, "nan");
    } else {
      strcpy(out, "inf");
    }
    return {first + len, {}};
  }

  // Normalize number
  if (exp != 0) {
    // Normal number
    // Apply implicit leading 1
    mantissa |= (uint_t(1) << mantissa_bits);
    // Take implicit 1 into account for the exponent
    --exp;
  } else {
    if (!mantissa) {
      // Zeroes
      strcpy(out, "0.000000e+00");
      return {first + len, {}};
    }
    // Subnormals (no need to normalize)
  }

  BaseTwoExponent<FloatType, exp_precision> base;

  // Buffer that will contain the decimal representation of the result
  DecimalRepresentation<FloatType, precision> decimal;

  // Loop over each bit of the mantissa, add the corresponding power of 2 if
  // the bit is set
  for (uint64_t mask = 0x1; mask < (uint_t(1) << (mantissa_bits + 1));
       mask <<= 1, ++exp) {
    if (mantissa & mask) {
      base.generate_nth_exp(exp);

      // Shift the buffer if current power of 2 has a greater base 10 exponent
      // than the current computed number
      int shift = base.exp10 - decimal.exp10;
      if (shift > 0) {
        if (decimal.shift_right(0, shift) >= 5) {
          decimal.round_up();
        }
      }

      decimal.exp10 = base.exp10;

      // Add current power of two with number
      decimal += base;
    }
  }

  // Round to nearest, tie to even
  int need_round = false;
  if (decimal.buffer[7] >= 5) {
    if (decimal.buffer[7] == 5) {
      // In case the last digit is 5, we round up if:
      //  - we find a non zero digit after `buffer[7]`
      //  - we find no non zero digit after `buffer[7]` but `buffer[6]` is an
      //  odd number.
      //  This is the `tie to even` part of the rounding.
      //  (this is also where the lack of precision is causing this
      //  implementation to sometime output wrong results)
      bool all_zero = true;
      for (int i = 8; i < precision && all_zero; ++i) {
        all_zero = decimal.buffer[i] == 0;
      }
      need_round = !all_zero || ((decimal.buffer[6] % 2) == 1);
    } else {
      need_round = true;
    }
  }

  if (need_round) {
    decimal.round_up(6);
  }

  // Copy the significand to the output
  *out++ = decimal.buffer[0] + '0';
  *out++ = '.';
  for (int i = 1; i < 7; ++i) {
    *out++ = decimal.buffer[i] + '0';
  }

  // Write exponent
  *out++ = 'e';
  *out++ = decimal.exp10 >= 0 ? '+' : '-';

  unsigned int abs_exp = decimal.exp10 >= 0 ? decimal.exp10 : -decimal.exp10;
  int exp_len          = to_chars_len(abs_exp);
  // Add a leading '0' if exp length is smaller than 2
  if (exp_len < 2) {
    *out++ = '0';
  }

  to_chars_i(out, last, abs_exp);
  out += exp_len;

  return {first + len, {}};
}

}  // namespace Impl
}  // namespace Kokkos

#endif
