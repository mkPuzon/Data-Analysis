from data import Data

def test():
    filepath_iris = 'data\iris.csv'
    filepath_types = 'data\\test_data_complex.csv'
    filepath_miss = 'data\\test_data_missing.csv'
    filepath_miss2 = 'data\\test_missing.csv'
    
    test_data = Data(filepath_iris)
    
if __name__ == "__main__":
    test()