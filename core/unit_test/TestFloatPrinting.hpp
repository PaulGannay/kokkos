// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>

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
      if constexpr (std::is_same_v<FloatType, float>) {
        Kokkos::printf("For float %s\n", ref);
      } else {
        Kokkos::printf("For double %s\n", ref);
      }
      ++errors;
    }

    if (strcmp(buffer, ref) != 0) {
      Kokkos::printf("Error: %s != %s\n", buffer, ref);
      if constexpr (std::is_same_v<FloatType, float>) {
        Kokkos::printf("For float %s\n", ref);
      } else {
        Kokkos::printf("For double %s\n", ref);
      }
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

      // Numbers that were incorrectly displayed at some point during the
      // debugging process
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x0000080000000003),
                                  "4.345847e-311");
      errors += to_chars_helper_f(Kokkos::bit_cast<double>(0x00072800000002AF),
                                  "9.951990e-309");
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

#ifdef FULL_TEST
int check(double d) {
  char ref[30];
  char ref_string[30];
  sprintf(ref_string, "%%5.%ie", 7 - 1);

  double ref_number = d;
  sprintf(ref, ref_string, ref_number);

  bool ret = false;

  if (std::isnan(d) || std::isinf(d)) {
    return ret;
  }

  char buffer[25];
  using Kokkos::Impl::to_chars_f;
  to_chars_f(buffer, buffer + 25, d);

  for (int i = 0; i < 10; ++i) {
    if (ref[i] != buffer[i]) {
      printf("0x%lx\n", std::bit_cast<uint64_t>(d));
      printf("%s !=\n", ref);
      printf("%s\n", buffer);

      for (int j = 0; j < i; ++j) {
        printf(" ");
      }
      printf("^\n");
      ret = true;
      break;
    }
  }

  return ret;
}

void do_random_test() {
  Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> random_pool(/*seed=*/12345);

  // Don't check negative numbers, implementation is simple enough, bugs should
  // be caught by the standard tests
  uint64_t max = 0x1llu << 28;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, max),
      KOKKOS_LAMBDA(uint64_t counter) {
        auto generator = random_pool.get_state();

        double d = Kokkos::bit_cast<double>(generator.urand64());

        // do not forget to release the state of the engine
        random_pool.free_state(generator);
        ASSERT_EQ(check(d), 0);

        ++counter;

        if (counter % 1'000'000 == 0) {
          Kokkos::printf("%f%% - %e\n", (double)counter / (double)max * 100.,
                         d);
        }
      });
}
#endif

TEST(TEST_CATEGORY, FloatPrinting) {
  TestFloatPrinting<TEST_EXECSPACE, double>();
  TestFloatPrinting<TEST_EXECSPACE, float>();

#ifdef FULL_TEST
  do_random_test();
#endif
}
