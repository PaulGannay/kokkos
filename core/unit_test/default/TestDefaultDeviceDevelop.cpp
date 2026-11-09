// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>
#include "KokkosTest_Utils.hpp"
#include "../TestStringManipulationRuntime.hpp"

KOKKOS_DEVICE_TEST(TestToto, T) {
  T a = 0;
  T b = 128;
  KOKKOS_EXPECT_EQ(m_test_reporter, a, b);
}

template <typename T>
void test() {
  TestToto<TEST_EXECSPACE, T> test;
  ASSERT_EQ(test.run(), 0);
}

namespace Test {

TEST(defaultdevicetype, development_test) {
  test<int8_t>();
  //test<uint8_t>();
  //test<int16_t>();
  //test<uint16_t>();
  //test<int32_t>();
  //test<uint32_t>();
  //test<int64_t>();
  //test<uint64_t>();
  //test<float>();
  //test<double>();
}

}  // namespace Test
