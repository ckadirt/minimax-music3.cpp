#include "engine/community_models/minimax_music3/seed.h"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    using namespace engine::models::minimax_music3;

    assert(derive_ar_sampling_seed(0) == 480593764U);
    assert(derive_ar_sampling_seed(1) == 1838448569U);
    assert(derive_ar_sampling_seed(42) == 1398456984U);
    assert(derive_ar_sampling_seed(UINT64_MAX) == 1689010001U);

    assert(derive_dit_chunk_seed(0, 0) == 1172308609478234070ULL);
    assert(derive_dit_chunk_seed(0, 1) == 5227974074385879887ULL);
    assert(derive_dit_chunk_seed(42, 17) == 1268231586810140013ULL);
    assert(derive_dit_chunk_seed(UINT64_MAX, 1) == 5976228930309119511ULL);

    assert(murmur_sampling_hash(0, 0, 0) == 2167721464U);
    assert(murmur_sampling_hash(1398456984U, 0, 0) == 1643673188U);
    assert(murmur_sampling_hash(1398456984U, 0, 1) == 1839046818U);
    assert(murmur_sampling_hash(1398456984U, 7, 1023) == 3182922125U);
    assert(murmur_sampling_hash(0x123456789abcdef0ULL, UINT32_MAX, 42) == 2305752703U);

    const std::vector<float> scores{0.0F, 0.0F, 0.0F};
    const auto first = seeded_gumbel_argmax(scores, 1398456984U, 8);
    assert(first == seeded_gumbel_argmax(scores, 1398456984U, 8));
    return 0;
}
