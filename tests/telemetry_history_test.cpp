// /tests/telemetry_history_test.cpp

#include <cassert>
#include <iostream>

#include "aerial/telemetry/telemetry_history.hpp"

using aerial::telemetry::TelemetryHistory;

void TestInitialState()
{
    TelemetryHistory<double, 4> history;

    assert(history.Empty());
    assert(history.Size() == 0);
    assert(history.MaxSize() == 4);
}

void TestInsertionOrder()
{
    TelemetryHistory<int, 4> history;

    history.Push(10);
    history.Push(20);
    history.Push(30);

    assert(history.Size() == 3);

    assert(history.At(0) == 10);
    assert(history.At(1) == 20);
    assert(history.At(2) == 30);
}

void TestCircularOverwrite()
{
    TelemetryHistory<int, 4> history;

    for (int i = 1; i <= 6; ++i) {
        history.Push(i);
    }

    assert(history.Size() == 4);

    assert(history.At(0) == 3);
    assert(history.At(1) == 4);
    assert(history.At(2) == 5);
    assert(history.At(3) == 6);
}

void TestClear()
{
    TelemetryHistory<int, 4> history;

    history.Push(10);
    history.Push(20);

    history.Clear();

    assert(history.Empty());
    assert(history.Size() == 0);

    history.Push(42);

    assert(history.Size() == 1);
    assert(history.At(0) == 42);
}

void TestRepeatedWraparound()
{
    TelemetryHistory<int, 3> history;

    for (int i = 0; i < 100; ++i) {
        history.Push(i);
    }

    assert(history.Size() == 3);

    assert(history.At(0) == 97);
    assert(history.At(1) == 98);
    assert(history.At(2) == 99);
}

int main()
{
    TestInitialState();
    TestInsertionOrder();
    TestCircularOverwrite();
    TestClear();
    TestRepeatedWraparound();

    std::cout << "Telemetry history tests PASSED\n";

    return 0;
}
