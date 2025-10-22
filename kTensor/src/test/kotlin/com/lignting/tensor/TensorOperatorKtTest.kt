package com.lignting.tensor

import com.lignting.number.typeDouble
import com.lignting.number.typeInt
import org.junit.jupiter.api.Test

class TensorOperatorKtTest {
    @Test
    fun testPlus() {
        val int = 2
        val double = 2.0
        val intTensor = tensorIntOf(1, 2, 3)
        val doubleTensor = tensorDoubleOf(1.0, 2.0, 3.0)

        val intTensorPlusInt = intTensor + int
        val intTensorPlusDouble = intTensor + double
        val doubleTensorPlusDouble = doubleTensor + double
        val doubleTensorPlusInt = doubleTensor + int
        val intTensorPlusIntTensor = intTensor + intTensor
        val intTensorPlusDoubleTensor = intTensor + doubleTensor
        val doubleTensorPlusDoubleTensor = doubleTensor + doubleTensor
        val doubleTensorPlusIntTensor = doubleTensor + intTensor

        assert(intTensorPlusInt.data.contentEquals(arrayOf(3.typeInt(), 4.typeInt(), 5.typeInt())))
        assert(intTensorPlusDouble.data.contentEquals(arrayOf(3.0.typeDouble(), 4.0.typeDouble(), 5.0.typeDouble())))
        assert(doubleTensorPlusDouble.data.contentEquals(arrayOf(3.0.typeDouble(), 4.0.typeDouble(), 5.0.typeDouble())))
        assert(doubleTensorPlusInt.data.contentEquals(arrayOf(3.0.typeDouble(), 4.0.typeDouble(), 5.0.typeDouble())))
        assert(intTensorPlusIntTensor.data.contentEquals(arrayOf(2.typeInt(), 4.typeInt(), 6.typeInt())))
        assert(intTensorPlusDoubleTensor.data.contentEquals(arrayOf(2.0.typeDouble(), 4.0.typeDouble(), 6.0.typeDouble())))
        assert(doubleTensorPlusDoubleTensor.data.contentEquals(arrayOf(2.0.typeDouble(), 4.0.typeDouble(), 6.0.typeDouble())))
        assert(doubleTensorPlusIntTensor.data.contentEquals(arrayOf(2.0.typeDouble(), 4.0.typeDouble(), 6.0.typeDouble())))
    }

    @Test
    fun testPlusBoardCast() {
        val intTensor = tensorIntOf(1, 2, 3).reshape(3, 1)
        val doubleTensor = tensorDoubleOf(1.0, 2.0, 3.0).reshape(3, 1)
        val intTensor2 = tensorIntOf(1, 2).reshape(1, 2)
        val doubleTensor2 = tensorDoubleOf(1.0, 2.0).reshape(1, 2)

        val intTensorPlusIntTensor = intTensor + intTensor2
        val intTensorPlusDoubleTensor = intTensor + doubleTensor2
        val doubleTensorPlusDoubleTensor = doubleTensor + doubleTensor2
        val doubleTensorPlusIntTensor = doubleTensor + intTensor2

        assert(intTensorPlusIntTensor.data.contentEquals(arrayOf(2.typeInt(), 3.typeInt(), 3.typeInt(), 4.typeInt(), 4.typeInt(), 5.typeInt())))
        assert(intTensorPlusDoubleTensor.data.contentEquals(arrayOf(2.0.typeDouble(), 3.0.typeDouble(), 3.0.typeDouble(), 4.0.typeDouble(), 4.0.typeDouble(), 5.0.typeDouble())))
        assert(doubleTensorPlusDoubleTensor.data.contentEquals(arrayOf(2.0.typeDouble(), 3.0.typeDouble(), 3.0.typeDouble(), 4.0.typeDouble(), 4.0.typeDouble(), 5.0.typeDouble())))
        assert(doubleTensorPlusIntTensor.data.contentEquals(arrayOf(2.0.typeDouble(), 3.0.typeDouble(), 3.0.typeDouble(), 4.0.typeDouble(), 4.0.typeDouble(), 5.0.typeDouble())))
    }
}