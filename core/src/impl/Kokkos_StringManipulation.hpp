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
#include <Kokkos_Printf.hpp>

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

template <class FloatType,
          std::enable_if_t<std::is_same_v<FloatType, double> ||
                               std::is_same_v<FloatType, float>,
                           int> = 0>
KOKKOS_FUNCTION unsigned int to_chars_len(FloatType f) {
  using uint_t                = Kokkos::equivalent_int_t<FloatType>;
  constexpr int mantissa_bits = Kokkos::mantissa_bits_v<FloatType>;
  constexpr int exponent_bits = Kokkos::exponent_bits_v<FloatType>;

  constexpr uint_t exp_mask      = (uint_t(1) << exponent_bits) - 1;
  constexpr uint_t mantissa_mask = (uint_t(1) << mantissa_bits) - 1;

  uint_t u = Kokkos::bit_cast<uint_t>(f);

  // Extract sign and exponent
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
  // Buffer that contains the decimal representation of a number in scientific
  // notation
  uint8_t buffer[size];
  // Exponent of the decimal representation stored
  int exp10;
  // For instance, if the number stored is 2^10, `buffer` contains
  // "102400000..." and `exp10` is 3

  // rem is the part of the floating point number that is beyond the decimal
  // representation. It is always comprised between 0 and 1. For instance, if
  // size is 3, if the number stored is 2^-6 = 0.015625:
  //  - exp10 = -2
  //  - buffer = 156
  //  - rem = 0.25
  double rem;

  KOKKOS_FUNCTION constexpr DecimalRepresentation()
      : buffer{0}, exp10{0}, rem{0} {}

  // This can't be higher than 60 since we need to be able to store up to
  // (2^max_div - 1) * 9 in the remainder/carry, bigger exponent would overflow
  // an uint64_t
  static constexpr int max_div = 60;
  static constexpr int max_mul = 60;

  // Print buffer for debugging purpose
  KOKKOS_FUNCTION void print() const {
    char tmp[size + 2];
    tmp[0] = buffer[0] + '0';
    tmp[1] = '.';
    for (std::size_t i = 2; i < size + 1; ++i) {
      tmp[i] = buffer[i - 1] + '0';
    }
    tmp[size + 1] = '\0';
    Kokkos::printf("%se%+03i (%e)", tmp, exp10, rem);
  }

  // Round number up (equivalent to adding 10 ^ (exp10 - size))
  KOKKOS_FUNCTION void round_up() {
    bool carry = true;
    int i      = size - 1;

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
      shift_right(1, 1);
    }
  }

  // Shift the decimal representation in buffer `shift` time to the right,
  // inserting `insert` as the leading numbers.
  KOKKOS_FUNCTION void shift_right(uint8_t shift, int insert) {
    if (shift < 1) {
      // nothing to do
      return;
    }

    for (int i = size - 1; i >= 0; --i) {
      if (std::size_t(i) + shift >= size) {
        // Add discarded digit to the remainder
        rem += buffer[i];
        rem /= 10;
      }

      if (i >= shift) {
        buffer[i] = buffer[i - shift];
      } else {
        buffer[i] = insert;
      }
    }

    if (shift > size) {
      for (int i = size; i < shift; ++i) {
        rem /= 10;
      }
    }

    exp10 += shift;
  }

  // Divide decimal number by 2^exp
  KOKKOS_FUNCTION void divide_by_power_of_two(int exp) {
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
        } else {
          rem *= 10.;
          dividend += (uint64_t)rem;
          rem -= (double)((uint64_t)rem);
        }

        if (dividend != 0 && dividend >= divisor) {
          break;
        }

        if (first_pass) {
          --exp10;
        } else if (dividend != 0) {
          buffer[res_idx++] = 0;
        }
      } while (dividend != 0 && res_idx < size);

      first_pass = false;

      if (res_idx < size) {
        buffer[res_idx++] = dividend >> exp;
        dividend %= divisor;
      }
    } while (res_idx < size);

    rem = ((double)dividend + rem) / (double)divisor;
  }

  // Multiply decimal number by 2^exp
  KOKKOS_FUNCTION void multiply_by_power_of_two(int exp) {
    uint64_t multiplior = 0x1llu << exp;
    rem *= multiplior;
    uint64_t carry = rem;
    rem -= carry;

    for (int i = size - 1; i >= 0; --i) {
      uint64_t tmp = ((uint64_t)(buffer[i]) << exp) + carry;
      buffer[i]    = tmp % 10;
      carry        = tmp / 10;
    }

    // Continue adding the carry to the computed number
    while (carry > 0) {
      shift_right(1, carry % 10);
      carry /= 10;
    }
  }

  template <size_t size_r>
  KOKKOS_FUNCTION DecimalRepresentation<FloatType, size> &operator+=(
      DecimalRepresentation<FloatType, size_r> rhs) {
    // Shift operands to ensure both have the same exponent
    if (rhs.exp10 < exp10) {
      rhs.shift_right(exp10 - rhs.exp10, 0);
    } else if (exp10 < rhs.exp10) {
      shift_right(rhs.exp10 - exp10, 0);
    }

    rem += rhs.rem;
    int carry = rem;
    rem -= carry;

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
      shift_right(1, 1);
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

  KOKKOS_FUNCTION void print() const {
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

  using uint_t                = Kokkos::equivalent_int_t<FloatType>;
  constexpr int mantissa_bits = Kokkos::mantissa_bits_v<FloatType>;
  constexpr int exponent_bits = Kokkos::exponent_bits_v<FloatType>;

  constexpr uint_t exp_mask      = (uint_t(1) << exponent_bits) - 1;
  constexpr uint_t mantissa_mask = (uint_t(1) << mantissa_bits) - 1;

  // Number of decimal digits outputed
  constexpr int precision = 7;

  std::ptrdiff_t const len = to_chars_len(f);
  if (last - first < len) {
    return {last, errc::value_too_large};
  }
  uint_t u = Kokkos::bit_cast<uint_t>(f);

  // Extract sign, mantissa and exponent
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

  // Normalize number if needed (subnormals don't need normalization)
  if (exp != 0) {
    // Apply implicit leading 1
    mantissa |= (uint_t(1) << mantissa_bits);
    // Take implicit 1 into account for the exponent
    --exp;
  }

  BaseTwoExponent<FloatType, precision> base;

  // Buffer that will contain the decimal representation of the result
  DecimalRepresentation<FloatType, precision> decimal;

  bool found_set_bit = false;
  // Loop over each bit of the mantissa, add the corresponding power of 2 if
  // the bit is set
  for (uint64_t mask = 0x1; mask < (uint_t(1) << (mantissa_bits + 1));
       mask <<= 1, ++exp) {
    if (mantissa & mask) {
      base.generate_nth_exp(exp);

      if (!found_set_bit) {
        decimal       = base;
        found_set_bit = true;
      } else {
        // Add current power of two with number
        decimal += base;
      }
    }
  }

  // Round to nearest, tie to even
  int need_round = false;
  if (decimal.rem >= .5) {
    if (decimal.rem == .5) {
      // In case the remainder is exactly 0.5, we round up if
      // `buffer[6]` is an  odd number.
      // This is the `tie to even` part of the rounding.
      need_round = (decimal.buffer[precision - 1] % 2) == 1;
    } else {
      need_round = true;
    }
  }

  if (need_round) {
    decimal.round_up();
  }

  // Copy the significand to the output
  *out++ = decimal.buffer[0] + '0';
  *out++ = '.';
  for (int i = 1; i < precision; ++i) {
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
