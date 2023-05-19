
def print_help():

    BOLD = '\033[1m'
    ITALICS = '\x1B[3m'
    END = '\033[0m'
    print("""ALPHAZERO TICTACTOE
 
ARGUMENTS
    """+BOLD+"run.py play"+END+ITALICS+" total_iterations threads"""+END+""" [model_name (leave blank if not using machine learning)]"""+END+BOLD+"""
    run.py train"""+END+ITALICS+""" iterations_per_training_game simultaneous_games threads memory_size model_name"""+ END+""" [self_learn=true]
          """)
if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()

    if len(sys.argv)==1:
        print_help()        
        sys.exit(0)
    if sys.argv[1]=="train":
        try:
            iterations = int(sys.argv[2])
            games_at_once= int(sys.argv[3])
            threads= int(sys.argv[4])
            memory_size = int(sys.argv[5])
            model_name =(sys.argv[6])
            self_learn = True
            if len(sys.argv)>=8:
                self_learn=False if sys.argv[7].strip().lower()=="false" else True
        except:
            print_help()
            raise Exception("FAILED TO PARSE ARGUMENTS.")
        import tictactoeai
        tictactoeai.train(self_learn,iterations,games_at_once,threads,memory_size,model_name)

    elif sys.argv[1]=="play":

        try:
            total_iterations = int(sys.argv[2])
            threads = int(sys.argv[3])
            use_nn = False
            model_name = ""
            if len(sys.argv)>=5:
                use_nn = True
                model_name = sys.argv[4]
        except:
            print_help()
            raise Exception("FAILED TO PARSE ARGUMENTS.")
        import tictactoeai
        tictactoeai.play(total_iterations,threads,use_nn,model_name)
    else: 
        print_help()
        print("FAILED TO PARSE ARGUMENTS. NEED TO USE EITHER PLAY OR TRAIN MODE.")

    sys.exit(0)
