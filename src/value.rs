use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

struct InternalValue {
    value: f32,
    grad: f32,
    backward: Option<Box<dyn Fn(&Value)>>,
    op: Option<String>,
    previous: Option<Vec<Value>>,
}

pub struct Value(Rc<RefCell<InternalValue>>);

impl Debug for InternalValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("value", &self.value)
            .field("grad", &self.grad)
            .field("op", &self.op)
            .finish()
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::new(value, None, None, None)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // check if the two Rc pointers point to the same memory location
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl Value {
    pub fn new(
        value: f32,
        backward: Option<Box<dyn Fn(&Self)>>,
        op: Option<String>,
        previous: Option<Vec<Value>>,
    ) -> Value {
        let value = Rc::new(RefCell::new(InternalValue {
            value,
            grad: 0.0,
            backward,
            op,
            previous,
        }));
        Value(value)
    }

    pub fn value(&self) -> f32 {
        self.0.borrow().value
    }

    pub fn set_value(&self, value: f32) {
        self.0.borrow_mut().value = value;
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f32) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn print_graph(&self) {
        let mut stack = vec![(self.clone(), 0)]; // Include depth in the stack
        while let Some((v, depth)) = stack.pop() {
            let indent = "  ".repeat(depth);
            println!("{}{:?}", indent, v.0.borrow());
            if let Some(previous) = &v.0.borrow().previous {
                for p in previous {
                    stack.push((p.clone(), depth + 1)); // Increase depth for children
                }
            }
        }
    }

    pub fn back_prop(&self) {
        fn build_sort(v: &Value, sort: &mut Vec<Value>, visited: &mut HashSet<Value>) {
            if visited.contains(v) {
                return;
            }
            visited.insert(v.clone());
            if let Some(previous) = &v.0.borrow().previous {
                for p in previous {
                    build_sort(&p, sort, visited);
                }
            }
            sort.push(v.clone());
        }

        let mut sort = vec![];
        let mut visited = HashSet::new();
        build_sort(&self, &mut sort, &mut visited);

        self.0.borrow_mut().grad = 1.0;
        for v in sort.iter().rev() {
            if let Some(backward) = &v.0.borrow().backward {
                backward(v);
            }
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("value", &self.value())
            .field("grad", &self.grad())
            .field("op", &self.0.borrow().op)
            .finish()
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        let self_clone = self.0.clone();
        let other_clone = other.0.clone();
        let self_clone_previous = self.0.clone();
        let other_clone_previous = other.0.clone();

        Value::new(
            self.value() + other.value(),
            Some(Box::new(move |out: &Value| {
                self_clone.borrow_mut().grad += out.grad();
                other_clone.borrow_mut().grad += out.grad();
            })),
            Some("add".to_string()),
            Some(vec![
                Value(self_clone_previous),
                Value(other_clone_previous),
            ]),
        )
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        let self_clone = self.0.clone();
        let self_clone_previous = self.0.clone();

        Value::new(
            -self.value(),
            Some(Box::new(move |out: &Value| {
                self_clone.borrow_mut().grad -= out.grad();
            })),
            Some("neg".to_string()),
            Some(vec![Value(self_clone_previous)]),
        )
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        iter.fold(Value::from(0.0), |acc, x| &acc + &x)
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        let self_clone = self.0.clone();
        let other_clone = other.0.clone();
        let self_clone_previous = self.0.clone();
        let other_clone_previous = other.0.clone();

        Value::new(
            self.value() - other.value(),
            Some(Box::new(move |out: &Value| {
                self_clone.borrow_mut().grad += out.grad();
                other_clone.borrow_mut().grad -= out.grad();
            })),
            Some("sub".to_string()),
            Some(vec![
                Value(self_clone_previous),
                Value(other_clone_previous),
            ]),
        )
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let self_clone = self.0.clone();
        let other_clone = other.0.clone();
        let self_clone_previous = self.0.clone();
        let other_clone_previous = other.0.clone();

        Value::new(
            self.value() * other.value(),
            Some(Box::new(move |out: &Value| {
                let self_grad_increase = out.grad() * other_clone.borrow().value;
                let other_grad_increase = out.grad() * self_clone.borrow().value;
                self_clone.borrow_mut().grad += self_grad_increase;
                other_clone.borrow_mut().grad += other_grad_increase;
            })),
            Some("mul".to_string()),
            Some(vec![
                Value(self_clone_previous),
                Value(other_clone_previous),
            ]),
        )
    }
}

impl Value {
    pub fn pow(&self, n: u32) -> Value {
        let self_clone = self.0.clone();
        let self_clone_previous = self.0.clone();

        Value::new(
            self.value().powf(n as f32),
            Some(Box::new(move |out: &Value| {
                let grad_increase =
                    out.grad() * n as f32 * self_clone.borrow().value.powf(n as f32 - 1.0);
                self_clone.borrow_mut().grad += grad_increase;
            })),
            Some("pow".to_string()),
            Some(vec![Value(self_clone_previous)]),
        )
    }

    pub fn relu(&self) -> Value {
        let self_clone = self.0.clone();
        let self_clone_previous = self.0.clone();

        Value::new(
            self.value().max(0.0),
            Some(Box::new(move |out: &Value| {
                let grad_increase = out.grad()
                    * if self_clone.borrow().value > 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                self_clone.borrow_mut().grad += grad_increase;
            })),
            Some("relu".to_string()),
            Some(vec![Value(self_clone_previous)]),
        )
    }

    pub fn sigmoid(&self) -> Value {
        let self_clone = self.0.clone();
        let self_clone_previous = self.0.clone();

        Value::new(
            1.0 / (1.0 + (-self.value()).exp()),
            Some(Box::new(move |out: &Value| {
                let grad_increase = out.grad() * out.value() * (1.0 - out.value());
                self_clone.borrow_mut().grad += grad_increase;
            })),
            Some("sigmoid".to_string()),
            Some(vec![Value(self_clone_previous)]),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn add() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(3.0);
        let v3: Value = &v1 + &v2;

        v3.back_prop();

        assert_eq!(v3.value(), 5.0);
        assert_eq!(v1.grad(), 1.0);
        assert_eq!(v2.grad(), 1.0);
    }

    #[test]
    fn hash() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(2.0);
        dbg!(&v1.0);

        let mut hs = HashSet::new();
        dbg!("created");
        hs.insert(&v1);
        dbg!("inserted");
        hs.insert(&v2);
        assert_eq!(hs.len(), 2);
        hs.insert(&v1);
        assert_eq!(hs.len(), 2);
    }

    #[test]
    fn sub() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(3.0);
        let v3 = &v1 - &v2;
        v3.back_prop();

        assert_eq!(v3.value(), -1.0);
        assert_eq!(v1.grad(), 1.0);
        assert_eq!(v2.grad(), -1.0);
    }

    #[test]
    fn mul() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(3.0);
        let v3 = &v1 * &v2;
        v3.back_prop();

        assert_eq!(v3.value(), 6.0);
        assert_eq!(v1.grad(), 3.0);
        assert_eq!(v2.grad(), 2.0);
    }

    #[test]
    fn pow() {
        let v1 = Value::from(2.0);
        let v3 = v1.pow(3);
        dbg!(&v3);
        v3.back_prop();
        dbg!(&v3);

        assert_eq!(v3.value(), 8.0);
        assert_eq!(v1.grad(), 12.0);
    }

    #[test]
    fn loss_fn() {
        let output = Value::from(2.0);
        let target = Value::from(3.0);

        let loss = (&output - &target).pow(2);
        loss.back_prop();

        dbg!(&loss);
        dbg!(&output);
        dbg!(&target);

        assert_eq!(loss.value(), 1.0);
        assert_eq!(target.grad(), 2.0);
        assert_eq!(output.grad(), -2.0);
    }

    #[test]
    fn relu() {
        let v1 = Value::from(0.2);
        let v2 = &v1 * &v1;
        let v3 = v2.relu();

        v3.back_prop();

        assert!(v3.value() - 0.04 < 1e-6);
        assert_eq!(v1.grad(), 0.4);
    }

    #[test]
    fn longer_chain() {
        let v1 = Value::from(2.0);
        let v2 = Value::from(3.0);
        let v3 = &v1 + &v2;
        let v4 = &v3 * &v1;
        let v5 = &v4.pow(2);
        v5.back_prop();

        assert_eq!(v5.value(), 100.0);
        assert_eq!(v1.grad(), 140.0);
        assert_eq!(v2.grad(), 40.0);
    }

    #[test]
    fn another_long_chain() {
        let x = Value::from(-4.0);
        let z = &(&(&x * &Value::from(2.0)) + &Value::from(2.0)) + &x;
        let q = &(z.relu()) + &(&z * &x);
        let h = (&z * &z).relu();
        let y = &(&h + &q) + &(&q * &x);
        y.back_prop();

        assert_eq!(x.value(), -4.0);
        assert_eq!(z.value(), -10.0);
        assert_eq!(y.value(), -20.0);

        assert_eq!(x.grad(), 46.0);
        assert_eq!(z.grad(), -8.0);
    }
}
