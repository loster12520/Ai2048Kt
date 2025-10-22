package com.lignting.tensor

import com.lignting.number.NumberTypes

class Tensor<T : NumberTypes>(
    val shape: IntArray,
    val data: Array<T>
)