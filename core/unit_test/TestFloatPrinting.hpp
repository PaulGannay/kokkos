// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

template <class Space, class FloatType>
struct TestFloatPrinting {
  TestFloatPrinting() { run(); }
  void run() const {
    int errors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Space>(0, 1), *this, errors);
    ASSERT_EQ(errors, 0);
  }

  KOKKOS_FUNCTION constexpr int to_chars_helper_f(FloatType val,
                                                  char const* ref) const {
    using Kokkos::Impl::strcmp;
    using Kokkos::Impl::strlen;
    using Kokkos::Impl::to_chars_f;
    constexpr int BUFFER_SIZE = 21;

    int errors = 0;
    char buffer[BUFFER_SIZE];
    char* ptr = to_chars_f(buffer, buffer + BUFFER_SIZE, val).ptr;
    *ptr      = '\0';
    if (buffer + strlen(ref) != ptr) {
      Kokkos::printf("Error: %lx != %lx\n", buffer + strlen(ref), ptr);
      Kokkos::printf("For float %s\n", ref);
      ++errors;
    }

    if (strcmp(buffer, ref) != 0) {
      Kokkos::printf("Error: %s != %s\n", buffer, ref);
      Kokkos::printf("For float %s\n", ref);
      ++errors;
    }

    return errors;
  }

  KOKKOS_FUNCTION void operator()(int, int& errors) const {
    errors += to_chars_helper_f(Kokkos::infinity_v<FloatType>, "inf");
    errors += to_chars_helper_f(-Kokkos::infinity_v<FloatType>, "-inf");
    errors += to_chars_helper_f(Kokkos::quiet_NaN_v<FloatType>, "nan");
    errors += to_chars_helper_f(-Kokkos::quiet_NaN_v<FloatType>, "-nan");
    errors += to_chars_helper_f(Kokkos::signaling_NaN_v<FloatType>, "nan");
    errors += to_chars_helper_f(-Kokkos::signaling_NaN_v<FloatType>, "-nan");
    errors += to_chars_helper_f(FloatType(0.), "0.000000e+00");
    errors += to_chars_helper_f(FloatType(-0.), "-0.000000e+00");
    errors += to_chars_helper_f(FloatType(1.), "1.000000e+00");
    errors += to_chars_helper_f(FloatType(-1.), "-1.000000e+00");
    errors += to_chars_helper_f(FloatType(1.2), "1.200000e+00");
    errors += to_chars_helper_f(FloatType(-1.2), "-1.200000e+00");
    errors += to_chars_helper_f(FloatType(5.), "5.000000e+00");
    errors += to_chars_helper_f(FloatType(50.), "5.000000e+01");
    errors += to_chars_helper_f(FloatType(0.5), "5.000000e-01");
    errors += to_chars_helper_f(FloatType(6e20), "6.000000e+20");
    errors += to_chars_helper_f(FloatType(6e37), "6.000000e+37");

    if constexpr (std::is_same_v<FloatType, double>) {
      errors += to_chars_helper_f(FloatType(1e100), "1.000000e+100");
      errors +=
          to_chars_helper_f(Kokkos::denorm_min_v<FloatType>, "4.940656e-324");
      errors +=
          to_chars_helper_f(-Kokkos::denorm_min_v<FloatType>, "-4.940656e-324");
      errors +=
          to_chars_helper_f(Kokkos::finite_max_v<FloatType>, "1.797693e+308");
      errors +=
          to_chars_helper_f(-Kokkos::finite_max_v<FloatType>, "-1.797693e+308");

      // Numbers just before or after the point were we go from 12 characters
      // to 13.
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x2b617f7d402b1835),
                                  "1.000000e-99");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x2b617f7d402b1834),
                                  "1.000000e-99");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x2b617f7d402b1833),
                                  "9.999999e-100");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x2b617f7d402b1832),
                                  "9.999999e-100");

      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x54b249ad163d7d24),
                                  "9.999999e+99");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x54b249ad163d7d25),
                                  "9.999999e+99");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x54b249ad163d7d26),
                                  "1.000000e+100");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x54b249ad163d7d27),
                                  "1.000000e+100");

      // Round to nearest, ties to even: 0x51c9bcdf962a0f9f is exactly
      // "9.9999995E85", it should round up)
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x51c9bcdf962a0f9e),
                                  "9.999999e+85");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x51c9bcdf962a0f9f),
                                  "1.000000e+86");
    }

    if constexpr (std::is_same_v<FloatType, float>) {
      errors +=
          to_chars_helper_f(Kokkos::denorm_min_v<FloatType>, "1.401298e-45");
      errors +=
          to_chars_helper_f(-Kokkos::denorm_min_v<FloatType>, "-1.401298e-45");
      errors +=
          to_chars_helper_f(Kokkos::finite_max_v<FloatType>, "3.402823e+38");
      errors +=
          to_chars_helper_f(-Kokkos::finite_max_v<FloatType>, "-3.402823e+38");
    }
  }
};

TEST(TEST_CATEGORY, FloatPrinting) {
  TestFloatPrinting<TEST_EXECSPACE, double>();
  TestFloatPrinting<TEST_EXECSPACE, float>();
}
