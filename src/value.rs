use std::{
    cell::RefCell,
    collections::HashSet,
    iter::Sum,
    ops::{Add, Mul},
    ptr::NonNull,
    rc::Rc,
};

pub struct Value {
    pub value: Rc<RefCell<f32>>,
    pub grad: Rc<RefCell<f32>>,
    backward: Option<Box<dyn Fn(&Self)>>,
    previous: Option<Rc<Vec<Value>>>,
}

fn build_sort<'a>(
    s: &'a Value,
    sort: &mut Vec<&'a Value>,
    visited: &mut HashSet<NonNull<&'a Value>>,
) {
    if visited.contains(&NonNull::from(&s)) {
        return;
    }
    visited.insert(NonNull::from(&s));
    if let Some(previous) = &s.previous {
        for p in previous.iter() {
            build_sort(p, sort, visited);
        }
    }
    sort.push(s);
}

impl Value {
    pub fn new(value: f32, backward: Option<Box<dyn Fn(&Self)>>) -> Value {
        Value {
            value: Rc::new(RefCell::new(value)),
            grad: Rc::new(RefCell::new(0.0)),
            backward,
            previous: None,
        }
    }

    pub fn back_prop(&mut self) {
        self.grad = Rc::new(RefCell::new(1.0));
        let mut sort: Vec<&Value> = vec![];
        let mut visited: HashSet<NonNull<&Value>> = HashSet::new();
        build_sort(self, &mut sort, &mut visited);

        for s in sort.iter().rev() {
            if let Some(backward) = &s.backward {
                backward(s);
            }
        }
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let self_value = self.value.clone();
        let other_value = other.value.clone();
        let self_grad = self.grad.clone();
        let other_grad = other.grad.clone();

        Value::new(
            *self.value.borrow() * *other.value.borrow(),
            Some(Box::new(move |out: &Value| {
                *self_grad.borrow_mut() += *out.grad.borrow() * *other_value.borrow();
                *other_grad.borrow_mut() += *out.grad.borrow() * *self_value.borrow();
            })),
        )
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Value {
        iter.fold(Value::new(0.0, None), |acc, x| &acc + &x)
    }
}

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        let self_grad = self.grad.clone();
        let other_grad = other.grad.clone();

        Value::new(
            *self.value.borrow() + *other.value.borrow(),
            Some(Box::new(move |out: &Value| {
                *self_grad.borrow_mut() += *out.grad.borrow();
                *other_grad.borrow_mut() += *out.grad.borrow();
            })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let v1 = Value::new(2.0, None);
        let v2 = Value::new(3.0, None);
        let mut v3 = &v1 + &v2;
        v3.back_prop();

        assert_eq!(v3.value.borrow().to_owned(), 5.0);
        assert_eq!(v1.grad.borrow().to_owned(), 1.0);
        assert_eq!(v2.grad.borrow().to_owned(), 1.0);
    }

    #[test]
    fn mul() {
        let v1 = Value::new(2.0, None);
        let v2 = Value::new(3.0, None);
        let mut v3 = &v1 * &v2;
        v3.back_prop();

        assert_eq!(v3.value.borrow().to_owned(), 6.0);
        assert_eq!(v1.grad.borrow().to_owned(), 3.0);
        assert_eq!(v2.grad.borrow().to_owned(), 2.0);
    }
}
