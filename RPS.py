# RPS.py
import math
import random

# BEATS[x] = le coup qui bat x
BEATS = {"R": "P", "P": "S", "S": "R"}

# Pour estimer risque/perte (utile quand on n'est pas encore sûr du bot)
WIN_AGAINST = {"R": "S", "P": "R", "S": "P"}   # mon coup -> coup adverse battu
LOSE_AGAINST = {"R": "P", "P": "S", "S": "R"}  # mon coup -> coup adverse qui me bat


def _predict_quincy(t: int) -> str:
    # Quincy joue: choices[counter % 5] avec counter qui démarre à 0 puis s'incrémente au début.
    # Donc au round t (t>=1): choices[t % 5]
    choices = ["R", "R", "P", "P", "S"]
    return choices[t % 5]


def _predict_kris(my_hist) -> str:
    # Kris joue le coup qui bat notre coup précédent (si vide, considère "R")
    prev = my_hist[-1] if my_hist else ""
    if prev == "":
        prev = "R"
    return BEATS[prev]


def _predict_mrugesh(my_hist) -> str:
    # Mrugesh bat notre coup le plus fréquent sur les 10 derniers "prev_opponent_play"
    # Son historique commence par '' au premier appel, puis nos coups précédents.
    hist = [""] + list(my_hist)
    last_ten = hist[-10:]
    most_frequent = max(set(last_ten), key=last_ten.count)
    if most_frequent == "":
        most_frequent = "S"
    return BEATS[most_frequent]


def _predict_abbey(my_hist) -> str:
    # Abbey fait un Markov d'ordre 1 sur NOS transitions (paires) avec un 'R' injecté au départ.
    seq = ["R"] + list(my_hist)  # correspond à son opponent_history après append du prev_opponent_play

    play_order = {
        "RR": 0, "RP": 0, "RS": 0,
        "PR": 0, "PP": 0, "PS": 0,
        "SR": 0, "SP": 0, "SS": 0,
    }

    # Compte les transitions observées dans seq
    for i in range(1, len(seq)):
        play_order[seq[i - 1] + seq[i]] += 1

    prev = seq[-1]
    potential = [prev + "R", prev + "P", prev + "S"]
    sub = {k: play_order[k] for k in potential}

    # Même tie-break que son code (ordre: R puis P puis S)
    prediction_pair = max(sub, key=sub.get)
    predicted_my_next = prediction_pair[-1]
    return BEATS[predicted_my_next]


def _choose_move_from_probs(probs):
    """
    probs: dict {"R":w, "P":w, "S":w}
    On choisit le coup qui maximise win/(win+lose) (les égalités sont "safe" car le test ignore les ties).
    """
    total = sum(probs.values())
    if total <= 0:
        return random.choice(["R", "P", "S"])

    p = {k: v / total for k, v in probs.items()}

    best_move = None
    best_ratio = -1.0
    best_win = -1.0

    for my in ["R", "P", "S"]:
        win = p[WIN_AGAINST[my]]
        lose = p[LOSE_AGAINST[my]]
        denom = win + lose
        ratio = (win / denom) if denom > 0 else 0.5

        if (ratio > best_ratio) or (abs(ratio - best_ratio) < 1e-12 and win > best_win):
            best_ratio = ratio
            best_win = win
            best_move = my

    return best_move


def player(prev_play, state={"round": 0, "opp_hist": [], "my_hist": [], "scores": None, "last_preds": None}):
    """
    Stratégie:
    - On simule les 4 bots (Quincy, Abbey, Kris, Mrugesh) pour prédire leur coup au round courant.
    - Au round suivant, on compare leur prédiction au coup effectivement joué => score de vraisemblance.
    - Quand on est confiant, on joue le contre parfait du bot le plus probable.
    - Sinon, on joue une réponse robuste basée sur un mélange pondéré des prédictions.
    """

    # init scores
    if state["scores"] is None:
        state["scores"] = {b: 0.0 for b in ["quincy", "abbey", "kris", "mrugesh"]}

    # Détection de nouvelle partie (chaque match recommence avec prev_play == "")
    if prev_play == "" and state["round"] > 0:
        state["round"] = 0
        state["opp_hist"].clear()
        state["my_hist"].clear()
        state["scores"] = {b: 0.0 for b in ["quincy", "abbey", "kris", "mrugesh"]}
        state["last_preds"] = None

    # Mise à jour des scores avec le coup adverse du round précédent
    if state["round"] > 0:
        state["opp_hist"].append(prev_play)
        if state["last_preds"] is not None:
            for b, pred in state["last_preds"].items():
                if pred == prev_play:
                    state["scores"][b] += 1.0
                else:
                    state["scores"][b] -= 0.35

    # Round courant (1-indexé)
    state["round"] += 1
    t = state["round"]
    my_hist = state["my_hist"]

    # Prédictions du coup adverse au round t
    preds = {
        "quincy": _predict_quincy(t),
        "abbey": _predict_abbey(my_hist),
        "kris": _predict_kris(my_hist),
        "mrugesh": _predict_mrugesh(my_hist),
    }

    # On stocke ces prédictions pour les scorer au prochain appel
    state["last_preds"] = preds

    # Bot le plus probable
    ranked = sorted(state["scores"].items(), key=lambda kv: kv[1], reverse=True)
    best_bot, best_score = ranked[0]
    second_score = ranked[1][1]

    # Si on est assez confiant, on joue le contre parfait
    confident = (t >= 6) and (best_score - second_score >= 2.5)
    if confident:
        my_move = BEATS[preds[best_bot]]  # bat le coup adverse prédit
    else:
        # Mélange pondéré (softmax stable) des prédictions pour éviter des pertes inutiles au début
        max_s = max(state["scores"].values())
        weights = {b: math.exp((state["scores"][b] - max_s) / 2.0) for b in preds}

        probs = {"R": 0.0, "P": 0.0, "S": 0.0}
        for b, pmove in preds.items():
            probs[pmove] += weights[b]

        my_move = _choose_move_from_probs(probs)

    my_hist.append(my_move)
    return my_move