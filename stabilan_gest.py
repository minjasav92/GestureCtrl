from collections import deque, Counter

BUFFER = deque(maxlen=10)
log_majority = []

def stabilan_gest(novi_gest):
    BUFFER.append(novi_gest)
    if len(BUFFER) < 10:
        return None
        
    najcesci, broj = Counter(BUFFER).most_common(1)[0]
    if broj >= 7:
        log_majority.append({'gest': najcesci, 'stabilnost': broj})
        BUFFER.clear()
        return najcesci
        
    return None