use std::{
    fmt::Debug, iter::repeat_with, marker::PhantomData, mem::MaybeUninit, num::NonZeroUsize, ptr,
};

use growth_factors::Two;

pub mod growth_factors;

pub trait GrowthFactor {
    /// Calculates a new capacity for a dynamic array like a Vec based on the old capacity
    #[must_use]
    fn new_capacity(old_capacity: usize) -> usize {
        NonZeroUsize::new(old_capacity)
            .map_or_else(Self::new_capacity_if_old_capacity_eq_zero, |n| {
                Self::new_capacity_if_old_capacity_gt_zero(n)
            })
    }

    fn new_capacity_if_old_capacity_gt_zero(n: NonZeroUsize) -> usize;

    #[must_use]
    fn new_capacity_if_old_capacity_eq_zero() -> usize {
        1
    }
}

pub struct Vec<T, G = Two>
where
    G: GrowthFactor,
{
    /// Contents of the Vec.
    /// Not every element is initialized, so accessing this directly is unsafe.
    items: Box<[MaybeUninit<T>]>,

    /// Length of the Vec.
    /// Necessary because the contents of `items` are not all necessarily initialized.
    len: usize,

    phantom: PhantomData<G>,
}

impl<T, G> Default for Vec<T, G>
where
    G: GrowthFactor,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, G> Clone for Vec<T, G>
where
    T: Copy,
    G: GrowthFactor,
{
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
            len: self.len,
            phantom: self.phantom,
        }
    }
}

impl<T, G> Debug for Vec<T, G>
where
    T: Clone + Default + Debug,
    G: GrowthFactor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}, Length: {}, Capacity: {} ",
            self.as_slice(),
            self.len,
            self.cap(),
        )
    }
}

impl<T, G> Vec<T, G>
where
    G: GrowthFactor,
{
    /// Removes all elements from the `Vec`
    pub fn clear(&mut self) {
        // I'm sure there's a better way to do this, but this works so I'm leaving it here.
        while self.pop().is_some() {}
    }

    /// I have no idea if this will behave nicely, but could be faster than `Vec::clear`.
    /// # Safety
    /// I dunno use at your own risk.
    pub fn clear_stupid(&mut self) {
        for e in &mut self.items {
            *e = MaybeUninit::uninit();
        }
    }
}

impl<T, G> Vec<T, G>
where
    G: GrowthFactor,
    T: Clone,
{
    /**
    Resizes the `Vec` in-place so that `len` is equal to `new_len`.

    If `new_len` is greater than `len`, the `Vec` is extended by the difference, with each additional slot filled with value. If `new_len` is less than `len`, the `Vec` is simply truncated.

    This method requires `T` to implement `Clone`, in order to be able to clone the passed value. If you need more flexibility (or want to rely on `Default` instead of `Clone`), use `Vec::resize_with`. If you only need to resize to a smaller size, use `Vec::truncate`.
    */
    pub fn resize(&mut self, new_len: usize, value: T) {
        match new_len.cmp(&self.len) {
            std::cmp::Ordering::Less => self.truncate(new_len),
            std::cmp::Ordering::Greater => {
                // extend by difference with `value` in the new slots.
                self.extend(std::iter::repeat_n(value, self.len() - new_len));
            }
            std::cmp::Ordering::Equal => {}
        }
    }
}

impl<T, G> Vec<T, G>
where
    G: GrowthFactor,
{
    /**
    Shortens the vector, keeping the first len elements and dropping the rest.

    If len is greater or equal to the vector’s current length, this has no effect.

    The drain method can emulate truncate, but causes the excess elements to be returned instead of dropped.

    Note that this method has no effect on the allocated capacity of the vector.
    */
    pub fn truncate(&mut self, len: usize) {
        for e in self.items.iter_mut().skip(len) {
            *e = MaybeUninit::uninit();
        }
    }
}

impl<T, G> Vec<T, G>
where
    T: Clone,
    G: GrowthFactor,
{
    /// Returns the `Vec`'s items as a Box slice.
    #[must_use]
    pub fn into_box_slice(self) -> Box<[T]> {
        self.as_slice().into()
    }

    /// Reallocates the Vec to a new Boxed array with a desired length (the new capacity of this Vec)
    fn realloc_to_desired_cap(&mut self, new_capacity: usize) {
        let mut empty_space: Box<[MaybeUninit<T>]> = repeat_with(MaybeUninit::uninit)
            .take(new_capacity)
            .collect();

        // Iterator over the items in the Vec.
        // Move all of these into the new Box slice
        for (i, e) in self.as_slice().iter().cloned().enumerate() {
            empty_space[i] = MaybeUninit::new(e);
        }

        // Swap the current box slice with the new, larger one.
        self.items = empty_space;
    }

    fn realloc_if_spare_cap_lt_n(&mut self, n: usize) {
        if self.spare_capacity() < n {
            self.realloc_to_desired_cap(n.next_power_of_two());
        }
    }

    /// Grows the size of Vec to fit more items.
    fn realloc(&mut self) {
        self.realloc_to_desired_cap(G::new_capacity(self.cap()));
    }

    /// Calls `self.reallocate()` if `self.len() >= self.capacity()`.
    /// Returns whether the Vec reallocated.
    /// # Guarantees
    /// The length will always be less than the capcity after this is called, so calling this twice in a row is useless.
    fn realloc_if_len_gte_cap(&mut self) -> bool {
        let len_gte_cap = self.len >= self.cap();
        if len_gte_cap {
            self.realloc();
        }
        len_gte_cap
    }

    /// Shrinks the Vec so that its capacity is the same as its length, if possible.
    pub fn shrink_to_fit(&mut self) {
        self.realloc_to_desired_cap(self.len);
    }

    /**
    Returns the uninitialized portion of the Vec.
    Reallocates if the length of the Vec is greater than its capacity.
    */
    unsafe fn uninint_slice(&mut self) -> &mut [MaybeUninit<T>] {
        self.realloc_if_len_gte_cap();
        unsafe { self.items.get_unchecked_mut(self.len..) }
    }

    /**
    Returns a mutable reference to the first uninitialized value in `self.items`.
    Mostly for implementing methods like `self.push`.
    Reallocates if the length of the Vec is greater than its capacity.
    */
    unsafe fn first_uninit(&mut self) -> &mut T {
        unsafe {
            let uninit = self.uninint_slice();
            let e = uninit.get_unchecked_mut(0);
            &mut *ptr::from_mut::<MaybeUninit<T>>(e).cast::<T>()
        }
    }

    /**
    Pushes an item onto the end of the Vec.
    Reallocates if the length of the Vec is greater than its capacity.
    */
    pub fn push(&mut self, x: T) {
        *unsafe { self.first_uninit() } = x;
        self.len += 1;
    }

    /**
    Pushes all the items from an iterator into the Vec.
    May reallocate to fit all the items produced by the iterator.
    */
    pub fn extend<I>(&mut self, iter: I)
    where
        I: ExactSizeIterator + Iterator<Item = T>,
    {
        let len = iter.len();
        self.realloc_if_spare_cap_lt_n(len);

        /*
        Push `min_items` items from the iterator w/out reallocating.
        We know we have at least that much spare capacity.
        The iterator may have some spare items though, which we have to push the slow way.
        */
        unsafe { self.extend_unchecked(iter) };
    }

    /**
    Pushes all the items from an iterator into the Vec.
    May reallocate to fit all the items produced by the iterator.
    # Notes
    - In general you should call `Vec::extend` if you can, since that doesn't push one item at a time.
    - This should only be called when the number of remaining items in the iterator is unknown.
    */
    pub fn extend_naive(&mut self, iter: impl Iterator<Item = T>) {
        for x in iter {
            self.push(x);
        }
    }
}

impl<T, G> Vec<T, G>
where
    G: GrowthFactor,
{
    /// Creates a new empty Vec
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Box::new([]),
            len: 0,
            phantom: PhantomData,
        }
    }

    /// Returns a slice of all the items in the Vec.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let slice = &self.items[..self.len];
        unsafe { &*(ptr::from_ref::<[MaybeUninit<T>]>(slice) as *const [T]) }
    }
    /// Returns a slice of all the items in the Vec.
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let slice = &mut self.items[..self.len];
        unsafe { &mut *(ptr::from_mut::<[MaybeUninit<T>]>(slice) as *mut [T]) }
    }

    /// Returns a reference to an item in the Vec if its exists.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    /// Returns a mutable reference to an item in the Vec if its exists.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        // (index < self.len).then_some(unsafe { self.get_unchecked_mut(index) })
        self.as_slice_mut().get_mut(index)
    }

    /// Returns a reference to an item in the Vec without checking that it exists.
    /// # Safety
    /// `n` must be < `self.capacity()`
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.as_slice().get_unchecked(index) }
    }

    /// Returns a mutable reference to an item in the Vec without checking that it exists.
    /// # Safety
    /// `n` must be < `self.capacity()`
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe { self.as_slice_mut().get_unchecked_mut(index) }
    }

    /// Pushes an item into the Vec without reallocating
    /// # Safety
    /// The spare capacity of the Vec must be non-zero
    pub unsafe fn push_unchecked(&mut self, x: T) {
        self.len += 1;
        *unsafe { self.last_unchecked_mut() } = x;
    }

    /// Pushes all the items from an iterator into the Vec.
    /// # Safety
    /// `self.spare_capacity()` must be <= the number of items in the iterator.
    pub unsafe fn extend_unchecked(&mut self, iter: impl Iterator<Item = T>) {
        for x in iter {
            unsafe { self.push_unchecked(x) };
        }
    }

    /// Returns a reference to the last element in the Vec if there is one.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        self.get(self.len.checked_sub(1)?)
    }

    /// Returns a mutable reference to the last element in the Vec if there is one.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.get_mut(self.len.checked_sub(1)?)
    }

    /**
    Returns a reference to the last element in the Vec if there is one.
    # Safety
    - self.len must be known to be non-zero.
    - the Vec must be known to be non-empty
    */
    #[must_use]
    pub unsafe fn last_unchecked(&self) -> &T {
        unsafe { self.get_unchecked(self.len.unchecked_sub(1)) }
    }

    /**
    Returns a mutable reference to the last element in the Vec if there is one.
    # Safety
    The Vec must be known to be non-empty
    */
    pub unsafe fn last_unchecked_mut(&mut self) -> &mut T {
        unsafe { self.get_unchecked_mut(self.len.unchecked_sub(1)) }
    }

    /// Removes and returns the last element in the Vec
    pub fn pop(&mut self) -> Option<T> {
        let old = std::mem::replace(
            self.items.get_mut(self.len.checked_sub(1)?)?,
            MaybeUninit::uninit(),
        );
        self.len -= 1;
        // We can assume this is initialized because self.items[self.len-1] should already be initialized before this function is even called.
        Some(unsafe { old.assume_init() })
    }

    /// Capacity of the Vec, or the number of items the Vec can store without reallocating.
    #[must_use]
    pub const fn cap(&self) -> usize {
        self.items.len()
    }

    /// Returns the number of additional items the Vec can store without reallocating
    #[must_use]
    pub const fn spare_capacity(&self) -> usize {
        self.cap() - self.len()
    }

    /// Returns the number of items the Vec is currently storing.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the length of the Vec is zero.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
