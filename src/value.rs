use std::{cell::RefCell, fmt::Debug, ops::Add, rc::Rc};

struct InternalValue {
    value: f32,
    grad: f32,
    backward: Option<Box<dyn Fn(&Value)>>,
    op: Option<String>,
    self_rc: Option<Value>,
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
            self_rc: None,
            previous,
        }));
        value.borrow_mut().self_rc = Some(Value(value.clone()));
        Value(value)
    }

    pub fn value(&self) -> f32 {
        self.0.borrow().value
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    pub fn back_prop(&self) {
        self.0.borrow_mut().grad = 1.0;
        let mut stack = vec![self.clone()];
        dbg!(&stack);
        while let Some(v) = stack.pop() {
            if let Some(backward) = &v.0.borrow().backward {
                backward(&v);
            }
            if let Some(previous) = &v.0.borrow().previous {
                for p in previous {
                    stack.push(p.clone());
                }
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
            .field("previous", &self.0.borrow().previous)
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
        let self_clone_backward = self.0.clone();
        let other_clone_backward = other.0.clone();
        let self_clone_previous = self.0.clone();
        let other_clone_previous = other.0.clone();

        Value::new(
            self.value() + other.value(),
            Some(Box::new(move |out: &Value| {
                self_clone_backward.borrow_mut().grad += out.grad();
                other_clone_backward.borrow_mut().grad += out.grad();
            })),
            Some("add".to_string()),
            Some(vec![
                Value(self_clone_previous),
                Value(other_clone_previous),
            ]),
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

    // #[test]
    // fn sub() {
    //     let v1 = InternalValue::from(2.0);
    //     let v2 = InternalValue::from(3.0);
    //     let mut v3 = &v1 - &v2;
    //     v3.back_prop();

    //     assert_eq!(v3.value.borrow().to_owned(), -1.0);
    //     assert_eq!(v1.grad.borrow().to_owned(), 1.0);
    //     assert_eq!(v2.grad.borrow().to_owned(), -1.0);
    // }

    // #[test]
    // fn pow() {
    //     let v1 = InternalValue::from(2.0);
    //     let mut v3 = v1.pow(3);
    //     v3.back_prop();

    //     assert_eq!(v3.value.borrow().to_owned(), 8.0);
    //     assert_eq!(v1.grad.borrow().to_owned(), 12.0);
    // }

    // #[test]
    // fn loss_fn() {
    //     let output = InternalValue::from(2.0);
    //     let target = InternalValue::from(3.0);

    //     let mut loss = (&output - &target).pow(2);
    //     loss.back_prop();

    //     dbg!(&loss);
    //     dbg!(&output);
    //     dbg!(&target);

    //     assert_eq!(*loss.value.borrow(), 1.0);
    //     assert_eq!(*target.grad.borrow(), 2.0);
    //     assert_eq!(*output.grad.borrow(), -2.0);
    // }

    // #[test]
    // fn mul() {
    //     let v1 = InternalValue::from(2.0);
    //     let v2 = InternalValue::from(3.0);
    //     let mut v3 = &v1 * &v2;
    //     v3.back_prop();

    //     assert_eq!(v3.value.borrow().to_owned(), 6.0);
    //     assert_eq!(v1.grad.borrow().to_owned(), 3.0);
    //     assert_eq!(v2.grad.borrow().to_owned(), 2.0);
    // }
}
