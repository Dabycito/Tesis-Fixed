import re
class Mutations:
    def run(self, mutation, sequence):
        if "delins" in mutation:
            return self.__parse_delins(mutation, sequence)
        if "del" in mutation:
            return self.__parse_deletion(mutation, sequence)
        if "ins" in mutation:
            return self.__parse_insertion(mutation, sequence)
        if "X" == mutation[-1] or "*" == mutation[-1]:
            return self.__parse_nonsense(mutation, sequence)
        if "=" == mutation[-1]:
            return self.__parse_silent(mutation, sequence)
        return self.__parse_missense(mutation, sequence)
        
    def __parse_delins(self, mutation, seq):
        pos, ins = mutation.split("delins")
        pos = re.findall(r'\d+', mutation)
        pos = [int(a) for a in pos]
        pos.sort()
        if len(pos) > 1:
            pos = list(range(pos[0], pos[1]+1))
        try:
            return seq[:pos[0]-1] + ins + seq[pos[-1]:]
        except:
            return None

    def __parse_deletion(self, mutation, seq):
        pos = re.findall(r'\d+', mutation)
        pos = [int(a) for a in pos]
        if len(pos) > 1:
            pos = list(range(pos[0], pos[1]+1))
        try:
            return seq[:pos[0]-1] + seq[pos[-1]:]
        except:
            return None

    def __parse_insertion(self, mutation, seq):
        mutation, ins = mutation.split("ins")
        pos = re.findall(r'\d+', mutation)
        pos = [int(a) for a in pos]
        pos.sort()
        if len(pos) > 1:
            pos = list(range(pos[0], pos[1]+1))
        try:
            return seq[:pos[-1]-1] + ins + seq[pos[-1]-1:]
        except:
            return None

    def __parse_nonsense(self, mutation, seq):
        pos = re.findall(r'\d+', mutation)[0]
        wt = mutation[mutation.index(pos[0]) - 1]
        pos = int(pos)
        try:
            if seq[pos-1] == wt:
                return seq[:pos-1]
        except:
            return None

    def __parse_silent(self, mutation, seq):
        return seq

    def __parse_missense(self, mutation, seq):
        pos = re.findall(r'\d+', mutation)[0]
        wt = mutation[mutation.index(pos[0]) - 1]
        pos = int(pos)
        mut = mutation[-1]
        try:
            if seq[pos-1] == wt:
                return seq[:pos-1] + mut + seq[pos:]
        except:
            return None

