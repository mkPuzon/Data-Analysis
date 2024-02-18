from data import Data

def test():
    filepath_iris = 'data\iris.csv'
    filepath_types = 'data\\test_data_complex.csv'
    
    test_data = Data(filepath_types)
    
if __name__ == "__main__":
    test()