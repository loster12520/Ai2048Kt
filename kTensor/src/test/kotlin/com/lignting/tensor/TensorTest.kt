package com.lignting.tensor

import com.lignting.number.IntNumber
import com.lignting.number.typeInt
import org.junit.jupiter.api.Assertions.*
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.test.Test

class TensorTest {
    @Test
    fun testTensor() {
        val tensor = tensorIntOf(1, 2, 3, 4, 5, 6)
        assertArrayEquals(intArrayOf(6), tensor.shape)
        assertEquals(6, tensor.data.size)
        assertEquals(1, tensor.data[0].toInt())
        assertEquals(2, tensor.data[1].toInt())
        assertEquals(3, tensor.data[2].toInt())
        assertEquals(4, tensor.data[3].toInt())
        assertEquals(5, tensor.data[4].toInt())
        assertEquals(6, tensor.data[5].toInt())

        val reshaped = tensor.reshape(2, 3)
        assertArrayEquals(intArrayOf(2, 3), reshaped.shape)
        assertEquals(6, reshaped.data.size)
        assertEquals(1, reshaped.data[0].toInt())
        assertEquals(2, reshaped.data[1].toInt())
        assertEquals(3, reshaped.data[2].toInt())
        assertEquals(4, reshaped.data[3].toInt())
        assertEquals(5, reshaped.data[4].toInt())
        assertEquals(6, reshaped.data[5].toInt())

        val expanded0 = reshaped.expand(0)
        assertEquals(2, expanded0.size) {
            "Expected size 2, but got ${expanded0.size}, in ${expanded0.map { it.shape.toList() to it.data.toList() }} }"
        }
        println("expanded0: ${expanded0.map { it.shape.toList() to it.data.toList().map { it.toInt() } }}")
        assertEquals(3, expanded0[0].shape[0])
        assertEquals(1, expanded0[0].data[0].toInt())
        assertEquals(2, expanded0[0].data[1].toInt())
        assertEquals(3, expanded0[0].data[2].toInt())
        assertEquals(3, expanded0[1].shape[0])
        assertEquals(4, expanded0[1].data[0].toInt())
        assertEquals(5, expanded0[1].data[1].toInt())
        assertEquals(6, expanded0[1].data[2].toInt())

        val expanded1 = reshaped.expand(1)
        println("expanded1: ${expanded1.map { it.shape.toList() to it.data.toList().map { it.toInt() } }}")
        assertEquals(3, expanded1.size)
        assertEquals(2, expanded1[0].shape[0])
        assertEquals(1, expanded1[0].data[0].toInt())
        assertEquals(4, expanded1[0].data[1].toInt())
        assertEquals(2, expanded1[1].shape[0])
        assertEquals(2, expanded1[1].data[0].toInt())
        assertEquals(5, expanded1[1].data[1].toInt())
        assertEquals(2, expanded1[2].shape[0])
        assertEquals(3, expanded1[2].data[0].toInt())
        assertEquals(6, expanded1[2].data[1].toInt())

        val expandedInvalid = assertThrows(IllegalArgumentException::class.java) {
            reshaped.expand(2)
        }
        assertEquals("dimension 2 out of range for shape [2, 3]", expandedInvalid.message)
    }

    @Test
    fun largeTensorTest() {
        val shape = IntArray(4) { 2 }
        val size = shape.reduce { acc, i -> acc * i }
        val data = Array(size) { IntNumber(it) }
        val tensor = Tensor(shape, data)

        assertArrayEquals(shape, tensor.shape)
        assertEquals(size, tensor.data.size)
        data.mapIndexed { index, value ->
            assertEquals(value.toInt(), tensor.data[index].toInt())
        }

        println("Tensor shape: ${shape.contentToString()}")
        println("Tensor data: ${tensor.data.map { it.toInt() }}")
        shape.mapIndexed { index, _ ->
            tensor.expand(index)
        }.forEachIndexed { index, value ->
            println("expand $index result: ${value.map { it.data.map { it.toInt() } }}")
        }
    }

    @Test
    fun broadcastOne() {
        val tensor = tensorIntOf(1, 2, 3, 4, 5, 6).reshape(3, 1, 2)
        val broadCasted = tensor.broadcastOne(1, 4)
        assertArrayEquals(intArrayOf(3, 4, 2), broadCasted.shape)
        println("broadCasted: ${broadCasted.data.map { it.toInt() }}")
        val expectedData = arrayOf(1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6)
        assertEquals(expectedData.size, broadCasted.data.size)
        expectedData.mapIndexed { index, value ->
            assertEquals(value, broadCasted.data[index].toInt())
        }
    }

    @Test
    fun broadcastAll() {
        val shape = intArrayOf(1, 2, 1, 3, 1)
        val tensor = Tensor(
            shape = shape,
            data = Array(shape.reduce { acc, i -> acc * i }) { it.typeInt() },
        )
        val targetShape = intArrayOf(2, 2, 3, 3, 2)
        val broadCasted = tensor.broadcast(targetShape)

        assertArrayEquals(targetShape, broadCasted.shape)
        println("broadCasted: ${broadCasted.data.map { it.toInt() }}")
        val expectedSize = targetShape.reduce { acc, i -> acc * i }
        assertEquals(expectedSize, broadCasted.data.size)
    }
}