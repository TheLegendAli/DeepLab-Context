#ifndef CPU_ONLY  // CPU-GPU test

#include <cstring>
#include <numeric>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/util/rank_element.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class RankElementTest : public ::testing::Test {};

TYPED_TEST_CASE(RankElementTest, TestDtypes);

TYPED_TEST(RankElementTest, TestRankElement) {
  Blob<TypeParam> blob(1, 1, 1, 100);
  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  filler.Fill(&blob);
  const TypeParam* valp = blob.cpu_data();
  const int count = blob.count();
  std::vector<TypeParam> valv(valp, valp + count);
  std::vector<int> rank;
  std::greater<TypeParam> comp;
  rank_element(rank, valv, comp);
  CHECK_EQ(rank.size(), count);
  const int sum = std::accumulate(rank.begin(), rank.end(), 0);
  CHECK_EQ(sum, count * (count - 1) / 2);
  for (int i = 0; i < rank.size() - 1; ++i) {
    if (i < 5) {
      LOG(INFO) << valp[rank[i]];
    }
    CHECK_GE(valp[rank[i]], valp[rank[i + 1]]);
  }
}

TYPED_TEST(RankElementTest, TestRankElementPartial) {
  Blob<TypeParam> blob(1, 1, 1, 100);
  const int top_k = 10;
  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  filler.Fill(&blob);
  const TypeParam* valp = blob.cpu_data();
  const int count = blob.count();
  std::vector<TypeParam> valv(valp, valp + count);
  std::vector<int> rank;
  std::greater<TypeParam> comp;
  partial_rank_element(rank, valv, top_k, comp);
  CHECK_EQ(rank.size(), top_k);
  for (int i = 0; i < top_k - 1; ++i) {
    if (i < 5) {
      LOG(INFO) << valp[rank[i]];
    }
    CHECK_GE(valp[rank[i]], valp[rank[i + 1]]);
  }
}

}  // namespace caffe

#endif  // CPU_ONLY
