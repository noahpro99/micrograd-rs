use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    hash::{Hash, Hasher},
    iter::Sum,
    ops::{Add, Mul, Sub},
    rc::Rc,
};

pub struct Value {
    pub value: Rc<RefCell<f32>>,
    pub grad: Rc<RefCell<f32>>,
    backward: Option<Box<dyn Fn(&Self)>>,
    op: Option<String>,
    self_rc: Option<Rc<Value>>,
    previous: Option<Vec<Rc<Value>>>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.value).hash(state);
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value {
            value: self.value.clone(),
            grad: self.grad.clone(),
            backward: None,
            op: self.op.clone(),
            self_rc: self.self_rc.clone(),
            previous: self.previous.clone(),
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("value", &self.value.borrow())
            .field("grad", &self.grad.borrow())
            .field("op", &self.op)
            .finish()
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::new(value, None, None, None)
    }
}

impl Value {
    pub fn new(
        value: f32,
        backward: Option<Box<dyn Fn(&Self)>>,
        op: Option<String>,
        previous: Option<Vec<Rc<Value>>>,
    ) -> Value {
        let mut value = Rc::new(Value {
            value: Rc::new(RefCell::new(value)),
            grad: Rc::new(RefCell::new(0.0)),
            backward,
            op,
            self_rc: None,
            previous,
        });
        let weak = Rc::downgrade(&value);
        Rc::get_mut(&mut value).unwrap().self_rc = Some(value.clone());

        return value;
    }

    pub fn relu(value: Value) -> Value {
        let value_clone = value.value.clone(); // Clone the value before moving it into the closure
        Value::new(
            match *value.value.borrow() {
                x if x > 0.0 => x,
                _ => 0.0,
            },
            Some(Box::new(move |out: &Value| {
                *value.grad.borrow_mut() += *out.grad.borrow()
                    * match *value_clone.borrow() {
                        x if x > 0.0 => 1.0,
                        _ => 0.0,
                    };
            })),
            Some("relu".to_string()),
            Some(vec![value.self_rc.unwrap().clone()]),
        )
    }

    pub fn sigmoid(value: Value) -> Value {
        Value::new(
            1.0 / (1.0 + (-*value.value.borrow()).exp()),
            Some(Box::new(move |out: &Value| {
                *value.grad.borrow_mut() +=
                    *out.grad.borrow() * *out.value.borrow() * (1.0 - *out.value.borrow());
            })),
            Some("sigmoid".to_string()),
            Some(vec![value.self_rc.unwrap().clone()]),
        )
    }

    pub fn pow(&self, n: i32) -> Value {
        let self_value = self.value.clone();
        let self_grad = self.grad.clone();

        Value::new(
            self.value.borrow().powi(n),
            Some(Box::new(move |out: &Value| {
                println!("running backward pow! with n: {}", n);
                println!("self_grad before: {}", *self_grad.borrow());
                *self_grad.borrow_mut() +=
                    *out.grad.borrow() * n as f32 * (*self_value.borrow()).powi(n - 1);
                println!("self_grad after: {}", *self_grad.borrow());
            })),
            Some("pow".to_string()),
            Some(vec![self.self_rc.clone().unwrap()]),
        )
    }

    pub fn back_prop(&mut self) {
        self.grad = Rc::new(RefCell::new(1.0));
        let mut sort: Vec<&Value> = vec![];
        let mut visited: HashSet<&Value> = HashSet::new();
        build_sort(Box::new(self), &mut sort, &mut visited);
        dbg!("all done sorting the graph!");
        dbg!(&sort);
        dbg!(&visited);

        for s in sort.iter().rev() {
            if let Some(backward) = &s.backward {
                backward(s);
            }
        }
    }
}

fn build_sort<'a>(s: Box<&'a Value>, sort: &mut Vec<&'a Value>, visited: &mut HashSet<&'a Value>) {
    if visited.contains(s.as_ref()) {
        return;
    }
    visited.insert(s.as_ref());
    if let Some(previous) = &s.previous {
        for p in previous.iter() {
            build_sort(Box::new(p), sort, visited);
        }
    }
    sort.push(s.as_ref());
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
            Some("mul".to_string()),
            Some(vec![
                self.self_rc.clone().unwrap(),
                other.self_rc.clone().unwrap(),
            ]),
        )
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Value {
        iter.fold(Value::from(0.), |acc, x| &acc + &x)
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
            Some("add".to_string()),
            Some(vec![
                self.self_rc.clone().unwrap(),
                other.self_rc.clone().unwrap(),
            ]),
        )
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        let self_grad = self.grad.clone();
        let other_grad = other.grad.clone();

        Value::new(
            *self.value.borrow() - *other.value.borrow(),
            Some(Box::new(move |out: &Value| {
                *self_grad.borrow_mut() += *out.grad.borrow();
                *other_grad.borrow_mut() -= *out.grad.borrow();
            })),
            Some("sub".to_string()),
            Some(vec![
                self.self_rc.clone().unwrap(),
                other.self_rc.clone().unwrap(),
            ]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(3.0);
        let mut v3 = &v1 + &v2;
        v3.back_prop();

        assert_eq!(*v3.value.borrow(), 5.0);
        assert_eq!(*v1.grad.borrow(), 1.0);
        assert_eq!(*v2.grad.borrow(), 1.0);
    }

    #[test]
    fn sub() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(3.0);
        let mut v3 = &v1 - &v2;
        v3.back_prop();

        assert_eq!(v3.value.borrow().to_owned(), -1.0);
        assert_eq!(v1.grad.borrow().to_owned(), 1.0);
        assert_eq!(v2.grad.borrow().to_owned(), -1.0);
    }

    #[test]
    fn pow() {
        let v1 = Value::from(2.0);
        let mut v3 = v1.pow(3);
        v3.back_prop();

        assert_eq!(v3.value.borrow().to_owned(), 8.0);
        assert_eq!(v1.grad.borrow().to_owned(), 12.0);
    }

    #[test]
    fn loss_fn() {
        let output = Value::from(2.0);
        let target = Value::from(3.0);

        let mut loss = (&output - &target).pow(2);
        loss.back_prop();

        dbg!(&loss);
        dbg!(&output);
        dbg!(&target);

        assert_eq!(*loss.value.borrow(), 1.0);
        assert_eq!(*target.grad.borrow(), 2.0);
        assert_eq!(*output.grad.borrow(), -2.0);
    }

    #[test]
    fn mul() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(3.0);
        let mut v3 = &v1 * &v2;
        v3.back_prop();

        assert_eq!(v3.value.borrow().to_owned(), 6.0);
        assert_eq!(v1.grad.borrow().to_owned(), 3.0);
        assert_eq!(v2.grad.borrow().to_owned(), 2.0);
    }
}
