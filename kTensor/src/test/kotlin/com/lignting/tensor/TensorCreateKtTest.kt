package com.lignting.tensor

import com.lignting.number.typeDouble
import com.lignting.number.typeInt
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*

class TensorCreateKtTest {

    @Test
    fun zeroIntTensor() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = zeroIntTensor(shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(0.typeInt(), value)
        }
    }

    @Test
    fun zeroI() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = zeroI(*shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(0.typeInt(), value)
        }
    }

    @Test
    fun onesIntTensor() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = onesIntTensor(shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(1.typeInt(), value)
        }
    }

    @Test
    fun onesI() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = onesI(*shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(1.typeInt(), value)
        }
    }

    @Test
    fun fullIntTensor() {
        val shape = intArrayOf(2, 3, 4)
        val fillValue = 7.typeInt()
        val tensor = fullIntTensor(shape, fillValue)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(fillValue, value)
        }
    }

    @Test
    fun fullI() {
        val shape = intArrayOf(2, 3, 4)
        val fillValue = 7.typeInt()
        val tensor = fullI(fillValue, *shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(fillValue, value)
        }
    }

    @Test
    fun zeroDoubleTensor() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = zeroDoubleTensor(shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(0.0.typeDouble(), value)
        }
    }

    @Test
    fun zeroD() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = zeroD(*shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(0.0.typeDouble(), value)
        }
    }

    @Test
    fun onesDoubleTensor() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = onesDoubleTensor(shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(1.0.typeDouble(), value)
        }
    }

    @Test
    fun onesD() {
        val shape = intArrayOf(2, 3, 4)
        val tensor = onesD(*shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(1.0.typeDouble(), value)
        }
    }

    @Test
    fun fullDoubleTensor(){
        val shape = intArrayOf(2, 3, 4)
        val fillValue = 7.0.typeDouble()
        val tensor = fullDoubleTensor(shape, fillValue)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(fillValue, value)
        }
    }

    @Test
    fun fullD(){
        val shape = intArrayOf(2, 3, 4)
        val fillValue = 7.0.typeDouble()
        val tensor = fullD(fillValue, *shape)
        assertArrayEquals(shape, tensor.shape)
        for (value in tensor.data) {
            assertEquals(fillValue, value)
        }
    }
}