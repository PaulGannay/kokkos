// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestSYCL_Category.hpp>
#include <TestMultiGPU.hpp>

namespace {

TEST(sycl_multi_gpu, managed_views) {
  auto execs = ::Kokkos::create_device_space();

  Kokkos::View<int *, TEST_EXECSPACE> view0(Kokkos::view_alloc("v0", execs[0]),
                                            100);
  Kokkos::View<int *, TEST_EXECSPACE> view1(Kokkos::view_alloc("v1", execs[1]),
                                            100);

  test_policies(execs[0], view0, execs[1], view1);
}

TEST(sycl_multi_gpu, unmanaged_views) {
  auto execs = ::Kokkos::create_device_space();

  int *p0 = sycl::malloc_device<int>(100, execs[0].sycl_queue());
  Kokkos::View<int *, TEST_EXECSPACE> view0(p0, 100);

  int *p1 = sycl::malloc_device<int>(100, execs[1].sycl_queue());
  Kokkos::View<int *, TEST_EXECSPACE> view1(p1, 100);

  test_policies(execs[0], view0, execs[1], view1);
  sycl::free(p0, execs[0].sycl_queue());
  sycl::free(p1, execs[1].sycl_queue());
}

TEST(sycl_multi_gpu, scratch_space) {
  auto execs = ::Kokkos::create_device_space();

  test_scratch(execs[0], execs[1]);
}
}  // namespace
