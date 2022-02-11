import os

module_num = int(input("Enter Module Number : "))

if module_num == 0:
    os.system("pytest tests/test_operators.py -k task0_1")
    os.system("pytest tests/test_operators.py -k task0_2")
    os.system("pytest tests/test_operators.py -k task0_3")
    os.system("pytest tests/test_module.py -k task0_4")

elif module_num == 1:
    os.system("pytest tests/test_scalar.py -k task1_1")
    os.system("pytest tests/test_scalar.py -k task1_2")
    os.system("pytest tests/test_autodiff.py -k task1_3")
    os.system("pytest tests/test_autodiff.py -k task1_4")

elif module_num == 2:
    os.system("pytest tests/test_tensor_data.py -k task2_1")
    os.system("pytest tests/test_tensor_data.py -k task2_2")
    os.system("pytest tests/test_tensor.py -k task2_3")
    os.system("pytest tests/test_tensor.py -k task2_4")

elif module_num == 3:
    os.system("pytest tests/test_tensor_general.py -k task3_1")
    os.system("pytest tests/test_tensor_general.py -k task3_2")
    print("Ran 2/4 cases here. Rest should be run on Google Colab.")

elif module_num == 4:
    os.system("pytest tests/test_conv.py -k task4_1")
    os.system("pytest tests/test_conv.py -k task4_2")
    os.system("pytest tests/test_nn.py -k task4_3")
    os.system("pytest tests/test_nn.py -k task4_4")

else:
    print("Module index must be in [0-4]")