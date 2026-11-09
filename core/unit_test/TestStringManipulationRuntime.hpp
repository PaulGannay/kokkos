// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

struct TestStaticString {
  TestStaticString() { EXPECT_EQ(run(), 0); }

  int run() const {
    int nerrors;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), *this,
                            nerrors);
    return nerrors;
  }

  template <std::size_t capacity>
  int check(const KokkosTest::UnformatedString<capacity>& us,
                            const char* ref) const {
    std::stringstream bs;
    bs << us;
    if (std::strcmp(bs.str().c_str(), ref) != 0) {
      Kokkos::printf("Error: %s != %s\n", bs.str().c_str(), ref);
      return 1;
    }
    return 0;
  }

  void operator()(int, int& errors) const {
    {
      // Empty constructor should return an empty valid cstring
      KokkosTest::UnformatedString<10> us;
      const char* ref = "";
      errors += check(us, ref);
    }
    {
      // Constructor taking a cstring
      KokkosTest::UnformatedString<10> us("test");
      const char* ref = "test";
      errors += check(us, ref);
    }
    {
      // Append cstring
      KokkosTest::UnformatedString<12> us("test");
      us << "test";
      const char* ref = "testtest";
      errors += check(us, ref);
    }
    {
      // Exceed string capacity in constructor
      KokkosTest::UnformatedString<10> us("testtesttest");
      const char* ref = "testtest...";
      errors += check(us, ref);
    }
    {
      // Append int
      KokkosTest::UnformatedString<20> us("test");
      us << 5;
      const char* ref = "test5";
      errors += check(us, ref);
    }
    {
      // Append several ints
      KokkosTest::UnformatedString<1024> us("test");
      us << 5 << 6 << 7;
      const char* ref = "test567";
      errors += check(us, ref);
    }
    //{
    //  // Completely fill the string without exceeding the capacity
    //  KokkosTest::UnformatedString<10> us("testtest");
    //  us << 5;
    //  const char* ref = "testtest5";
    //  errors += check(us, ref);
    //}
    //{
    //  // Fill the string and try to append an int
    //  KokkosTest::UnformatedString<10> us("testtestt");
    //  us << 5;
    //  const char* ref = "testte...";
    //  errors += check(us, ref);
    //}
    //{
    //  // Fill the string and try to append a cstring
    //  KokkosTest::UnformatedString<10> us("testtestt");
    //  us << "test";
    //  const char* ref = "testte...";
    //  errors += check(us, ref);
    //}
    //{
    //  // Try to write an int to big for the capacity, before appending a string
    //  KokkosTest::UnformatedString<10> us;
    //  us << 1'000'000'000 << "test";
    //  const char* ref = "...";
    //  errors += check(us, ref);
    //}
    //{
    //  // Multiple append
    //  KokkosTest::UnformatedString<250> us;
    //  us << "test" << 5 << "test" << 5 << "test";
    //  const char* ref = "test5test5test";
    //  errors += check(us, ref);
    //}
    //{
    //  // Append Booleans
    //  KokkosTest::UnformatedString<10> us;
    //  us << true << false;
    //  const char* ref = "TrueFalse";
    //  errors += check(us, ref);
    //}
    //{
    //  // Append Booleans without enough capacity
    //  KokkosTest::UnformatedString<10> us;
    //  us << "test" << true << false;
    //  const char* ref = "testTr...";
    //  errors += check(us, ref);
    //}
    //{
    //  // Append Float
    //  KokkosTest::UnformatedString<15> us;
    //  us << 1.5f;
    //  const char* ref = "1";
    //  errors += check(us, ref);
    //}
    //{
    //  // Append Double
    //  KokkosTest::UnformatedString<15> us;
    //  us << 1.5;
    //  const char* ref = "1";
    //  errors += check(us, ref);
    //}
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
    {
      // Append half_t
      KokkosTest::UnformatedString<20> us;
      us << Kokkos::Experimental::half_t(1.5f);
      const char* ref = "1";
      errors += check(us, ref);
    }
#endif
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
    {
      // Append half_t
      KokkosTest::UnformatedString<20> us;
      us << Kokkos::Experimental::bhalf_t(1.5f);
      const char* ref = "1";
      errors += check(us, ref);
    }
#endif
  }
};

TEST(TEST_CATEGORY, TestStaticString) { TestStaticString(); }

}  // namespace
