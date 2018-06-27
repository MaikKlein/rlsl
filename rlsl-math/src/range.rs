use std::iter::Iterator;
use std::ops::Range;
pub struct RlslRange(pub Range<u32>);

impl Iterator for RlslRange {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.start < self.0.end {
            let val = self.0.start;
            self.0.start += 1;
            Some(val)
        } else {
            None
        }
    }
}
