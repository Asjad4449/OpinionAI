from textblob import TextBlob

def read_arguments(file_path):
    pro_arguments = []
    con_arguments = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Pro:"):
                pro_arguments.append(line[4:].strip())
            elif line.startswith("Con:"):
                con_arguments.append(line[4:].strip())
    
    return pro_arguments, con_arguments

def analyze_sentiment(arguments):
    scores = [TextBlob(arg).sentiment.polarity for arg in arguments]
    return scores

def determine_winning_side(pro_scores, con_scores):
    avg_pro = sum(pro_scores) / len(pro_scores) if pro_scores else 0
    avg_con = sum(con_scores) / len(con_scores) if con_scores else 0
    
    if avg_pro > avg_con:
        return "Pro side has a stronger argument based on sentiment analysis."
    elif avg_con > avg_pro:
        return "Con side has a stronger argument based on sentiment analysis."
    else:
        return "Both sides are equally strong based on sentiment analysis."

def write_results(file_path, pro_scores, con_scores, result):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Pro Arguments Sentiment Scores:\n")
        file.writelines([f"{score}\n" for score in pro_scores])
        file.write("\nCon Arguments Sentiment Scores:\n")
        file.writelines([f"{score}\n" for score in con_scores])
        file.write("\nResult:\n" + result + "\n")

if __name__ == "__main__":
    input_file = "arg1.txt"
    output_file = "results.txt"
    
    pro_args, con_args = read_arguments(input_file)
    pro_scores = analyze_sentiment(pro_args)
    con_scores = analyze_sentiment(con_args)
    
    result = determine_winning_side(pro_scores, con_scores)
    write_results(output_file, pro_scores, con_scores, result)
