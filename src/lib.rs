use std::{fmt::Debug, iter::repeat_with, mem::MaybeUninit, ptr};

#[derive(Default)]
pub struct Vec<T> {
    /// Contents of the vector.
    /// Not every element is initialized, so accessing this directly is unsafe.
    items: Box<[MaybeUninit<T>]>,

    /// Length of the vector.
    /// Necessary because the contents of `items` are not all necessarily initialized.
    len: usize,
}

impl<T> Clone for Vec<T>
where
    T: Copy,
{
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
            len: self.len,
        }
    }
}

impl<T> Debug for Vec<T>
where
    T: Clone + Default + Debug,
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

impl<T> Vec<T>
where
    T: Clone,
{
    /// Returns the vector's items as a Box slice.
    #[must_use]
    pub fn into_box_slice(self) -> Box<[T]> {
        self.as_slice().into()
    }

    /// Reallocates the vector to a new Boxed array with a desired length (the new capacity of this vector)
    fn realloc_to_desired_cap(&mut self, new_capacity: usize) {
        let mut empty_space: Box<[MaybeUninit<T>]> = repeat_with(MaybeUninit::uninit)
            .take(new_capacity)
            .collect();

        // Iterator over the items in the vector.
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

    /// Grows the size of vector to fit more items.
    fn realloc(&mut self) {
        self.realloc_to_desired_cap((self.cap() + 1).next_power_of_two());
    }

    /// Calls `self.reallocate()` if `self.len() >= self.capacity()`.
    /// Returns whether the vector reallocated.
    /// # Guarantees
    /// The length will always be less than the capcity after this is called, so calling this twice in a row is useless.
    fn realloc_if_len_gte_cap(&mut self) -> bool {
        let len_gte_cap = self.len >= self.cap();
        if len_gte_cap {
            self.realloc();
            debug_assert!(self.len >= self.cap());
        }
        len_gte_cap
    }

    /// Shrinks the vector so that its capacity is the same as its length, if possible.
    pub fn shrink_to_fit(&mut self) {
        self.realloc_to_desired_cap(self.len);
    }

    /// Returns a mutable reference to the first uninitialized value in `self.items`.
    /// Mostly for implementing methods like `self.push`.
    /// - Reallocates if `self.length >= self.capacity`.    
    fn first_uninit(&mut self) -> &mut T {
        let s = self.uninint_slice();
        let e = s
            .get_mut(0)
            .expect("Expected at least one uninitialized element");
        unsafe { &mut *ptr::from_mut::<MaybeUninit<T>>(e).cast::<T>() }
    }

    /// Returns the uninitialized portion of the vector.
    fn uninint_slice(&mut self) -> &mut [MaybeUninit<T>] {
        self.realloc_if_len_gte_cap();
        self.items
            .get_mut(self.len..)
            .expect("Uninitialized portion of the vector")
    }

    /// Pushes an item onto the end of the vector
    pub fn push(&mut self, x: T) {
        *self.first_uninit() = x;
        self.len += 1;
    }

    /// Pushes all the items from an iterator into the vector
    pub fn extend(&mut self, mut iter: impl Iterator<Item = T>) {
        let min_items = iter.size_hint().0;
        self.realloc_if_spare_cap_lt_n(min_items);

        // Push `min_items` items from the iterator w/out reallocating.
        // We know we have at least that much spare capacity.
        // The iterator may have some spare items though, which we have to push the slow way.
        unsafe { self.extend_unchecked(iter.by_ref().take(min_items)) };
        self.extend_naive(iter);
    }

    /// Pushes all the items from an iterator into the vector.
    /// # Notes
    /// This should only be called when the number of remaining items in the iterator is unknown.
    pub fn extend_naive(&mut self, iter: impl Iterator<Item = T>) {
        iter.for_each(|x| {
            self.push(x);
        });
    }
}

impl<T> Vec<T> {
    /// Returns a slice of all the items in the vector.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let slice = &self.items[..self.len];
        unsafe { &*(ptr::from_ref::<[MaybeUninit<T>]>(slice) as *const [T]) }
    }
    /// Returns a slice of all the items in the vector.
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let slice = &mut self.items[..self.len];
        unsafe { &mut *(ptr::from_mut::<[MaybeUninit<T>]>(slice) as *mut [T]) }
    }

    /// Returns a reference to an item in the vector if its exists.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    /// Returns a mutable reference to an item in the vector if its exists.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        // (index < self.len).then_some(unsafe { self.get_unchecked_mut(index) })
        self.as_slice_mut().get_mut(index)
    }

    /// Returns a reference to an item in the vector without checking that it exists.
    /// # Safety
    /// `n` must be < `self.capacity()`
    #[must_use]
    pub const unsafe fn get_unchecked(&self, index: usize) -> &T {
        let item: &MaybeUninit<T> = &self.items[index];
        let ptr_mu_t: *const MaybeUninit<T> = ptr::from_ref(item);
        let ptr_t: *const T = ptr_mu_t.cast();
        &*ptr_t
    }

    /// Returns a mutable reference to an item in the vector without checking that it exists.
    /// # Safety
    /// `n` must be < `self.capacity()`
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.as_slice_mut().get_unchecked_mut(index)
    }

    /// Pushes an item into the vector without reallocating
    /// # Safety
    /// The spare capacity of the vector must be non-zero
    pub unsafe fn push_unchecked(&mut self, x: T) {
        self.len += 1;
        *self.last_unchecked_mut() = x;
    }

    /// Pushes all the items from an iterator into the vector.
    /// # Safety
    /// `self.spare_capacity()` must be <= the number of items in the iterator.
    pub unsafe fn extend_unchecked(&mut self, iter: impl Iterator<Item = T>) {
        for x in iter {
            self.push_unchecked(x);
        }
    }

    /// Returns a reference to the last element in the vector if there is one.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        self.get(self.len.checked_sub(1)?)
    }

    /// Returns a mutable reference to the last element in the vector if there is one.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.get_mut(self.len.checked_sub(1)?)
    }

    /// Returns a reference to the last element in the vector if there is one.
    /// # Safety
    /// - self.len must be known to be non-zero.
    /// - the vector must be known to be non-empty
    #[must_use]
    pub const unsafe fn last_unchecked(&self) -> &T {
        self.get_unchecked(self.len.unchecked_sub(1))
    }

    /// Returns a mutable reference to the last element in the vector if there is one.    
    /// # Safety
    /// The vector must be known to be non-empty
    pub unsafe fn last_unchecked_mut(&mut self) -> &mut T {
        self.get_unchecked_mut(self.len.unchecked_sub(1))
    }

    /// Removes and returns the last element in the vector
    pub fn pop(&mut self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        self.len -= 1;
        Some(unsafe { self.last_unchecked() })
    }

    /// Capacity of the vector, or the number of items the vector can store without reallocating.
    #[must_use]
    pub const fn cap(&self) -> usize {
        self.items.len()
    }

    /// Returns the number of additional items the vector can store without reallocating
    #[must_use]
    pub const fn spare_capacity(&self) -> usize {
        self.cap() - self.len()
    }

    /// Returns the number of items the vector is currently storing.

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the length of the vector is zero.    
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
