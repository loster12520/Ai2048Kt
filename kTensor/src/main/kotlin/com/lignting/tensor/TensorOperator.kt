package com.lignting.tensor

import com.lignting.number.DoubleNumber
import com.lignting.number.IntNumber
import com.lignting.number.typeDouble
import com.lignting.number.typeInt
import kotlin.collections.map

operator fun Tensor<IntNumber>.plus(other: Int): Tensor<IntNumber> =
    Tensor(
        shape = this.shape,
        data = this.data.map { it.toInt() + other }.map { it.typeInt() }.toTypedArray(),
    )

operator fun Tensor<DoubleNumber>.plus(other: Double): Tensor<DoubleNumber> =
    Tensor(
        shape = this.shape,
        data = this.data.map { it.toDouble() + other }.map { it.typeDouble() }.toTypedArray(),
    )