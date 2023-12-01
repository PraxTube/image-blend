# Image Blending

The `laplace_operator` method will give us the following
array for the input `5, 7`.

```
[[-4.  2.  0. ...  0.  0.  0.]
 [ 2. -4.  2. ...  0.  0.  0.]
 [ 0.  2. -4. ...  0.  0.  0.]
 ...
 [ 0.  0.  0. ... -4.  2.  0.]
 [ 0.  0.  0. ...  2. -4.  2.]
 [ 0.  0.  0. ...  0.  2. -4.]]
```

The shape of this array is `(5 * 7, 5 * 7) = (35, 35)`.
