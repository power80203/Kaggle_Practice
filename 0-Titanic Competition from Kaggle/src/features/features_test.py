
def test_listGenerator():

    test_array = list()

    for i in range(10):
        test_array.append(i)

    print(test_array)

    list_test = [i if i % 2 == 0 else "x" for i in test_array]

    print(list_test)

if __name__ == "__main__":
    test_listGenerator()