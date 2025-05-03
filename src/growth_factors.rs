use crate::GrowthFactor;

/// A growth factor of two.
/// Doubles the capacity on every reallocation.
pub struct Two;

impl GrowthFactor for Two {
    fn new_capacity_if_old_capacity_gt_zero(n: std::num::NonZeroUsize) -> usize {
        n.get() * 2
    }
}
