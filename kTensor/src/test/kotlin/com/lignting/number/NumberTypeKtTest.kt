package com.lignting.number

import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*

class NumberTypeKtTest {

    @Test
    fun type() {
        val int1 = IntNumber(5)
        val int2 = IntNumber(3)
        val double1 = DoubleNumber(5.0)
        val double2 = DoubleNumber(3.0)

        // IntNumber + IntNumber
        val intAdd = int1 + int2
        assertTrue(intAdd is IntNumber)
        assertEquals((intAdd as IntNumber).toInt(), 8)

        // IntNumber - IntNumber
        val intSub = int1 - int2
        assertTrue(intSub is IntNumber)
        assertEquals((intSub as IntNumber).toInt(), 2)

        // IntNumber * IntNumber
        val intMul = int1 * int2
        assertTrue(intMul is IntNumber)
        assertEquals((intMul as IntNumber).toInt(), 15)

        // IntNumber / IntNumber
        val intDiv = int1 / int2
        assertTrue(intDiv is IntNumber)
        assertEquals((intDiv as IntNumber).toInt(), 1)

        // DoubleNumber + DoubleNumber
        val doubleAdd = double1 + double2
        assertTrue(doubleAdd is DoubleNumber)
        assertEquals((doubleAdd as DoubleNumber).toDouble(), 8.0)

        // DoubleNumber - DoubleNumber
        val doubleSub = double1 - double2
        assertTrue(doubleSub is DoubleNumber)
        assertEquals((doubleSub as DoubleNumber).toDouble(), 2.0)

        // DoubleNumber * DoubleNumber
        val doubleMul = double1 * double2
        assertTrue(doubleMul is DoubleNumber)
        assertEquals((doubleMul as DoubleNumber).toDouble(), 15.0)

        // DoubleNumber / DoubleNumber
        val doubleDiv = double1 / double2
        assertTrue(doubleDiv is DoubleNumber)
        assertEquals((doubleDiv as DoubleNumber).toDouble(), 1.6666666666666667)

        // IntNumber + DoubleNumber
        val intDoubleAdd = int1 + double2
        assertTrue(intDoubleAdd is DoubleNumber)
        assertEquals((intDoubleAdd as DoubleNumber).toDouble(), 8.0)

        // IntNumber - DoubleNumber
        val intDoubleSub = int1 - double2
        assertTrue(intDoubleSub is DoubleNumber)
        assertEquals((intDoubleSub as DoubleNumber).toDouble(), 2.0)
    }

    @Test
    fun typeInt() {
        val int1 = 5.typeInt()
        val int2 = 3.typeInt()

        // IntNumber + IntNumber
        val intAdd = int1 + int2
        assertTrue(intAdd is IntNumber)
        assertEquals((intAdd as IntNumber).toInt(), 8)

        // IntNumber - IntNumber
        val intSub = int1 - int2
        assertTrue(intSub is IntNumber)
        assertEquals((intSub as IntNumber).toInt(), 2)

        // IntNumber * IntNumber
        val intMul = int1 * int2
        assertTrue(intMul is IntNumber)
        assertEquals((intMul as IntNumber).toInt(), 15)

        // IntNumber / IntNumber
        val intDiv = int1 / int2
        assertTrue(intDiv is IntNumber)
        assertEquals((intDiv as IntNumber).toInt(), 1)
    }

    @Test
    fun typeDouble(){
        val double1 = 5.0.typeDouble()
        val double2 = 3.0.typeDouble()

        // DoubleNumber + DoubleNumber
        val doubleAdd = double1 + double2
        assertTrue(doubleAdd is DoubleNumber)
        assertEquals((doubleAdd as DoubleNumber).toDouble(), 8.0)

        // DoubleNumber - DoubleNumber
        val doubleSub = double1 - double2
        assertTrue(doubleSub is DoubleNumber)
        assertEquals((doubleSub as DoubleNumber).toDouble(), 2.0)

        // DoubleNumber * DoubleNumber
        val doubleMul = double1 * double2
        assertTrue(doubleMul is DoubleNumber)
        assertEquals((doubleMul as DoubleNumber).toDouble(), 15.0)

        // DoubleNumber / DoubleNumber
        val doubleDiv = double1 / double2
        assertTrue(doubleDiv is DoubleNumber)
        assertEquals((doubleDiv as DoubleNumber).toDouble(), 1.6666666666666667)
    }
}