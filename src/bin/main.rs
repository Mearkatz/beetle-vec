use vec::Vec;

fn main() {
    let mut v: Vec<i32> = Vec::new();
    for n in 0..130 {
        println!("len = {}. cap = {}", v.len(), v.cap());
        v.push(n);
    }
}
