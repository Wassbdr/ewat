from .queries import Query
from .utils import *
import logging

logger = logging.getLogger("autoquery-scenario")
highspeed_weights = {True: 60, False: 40}

# Date future avec des trajets disponibles (les requêtes de billets l'exigent).
_PRESERVE_DATE = "2026-12-01"


def query_and_cancel(q: Query):
    if random_from_weighted(highspeed_weights):
        pairs = q.query_orders(types=tuple([0, 1]))
    else:
        pairs = q.query_orders(types=tuple([0, 1]), query_other=False)

    if not pairs:
        return

    # (orderId, tripId)
    pair = random_from_list(pairs)

    order_id = q.cancel_order(order_id=pair[0])
    if not order_id:
        return

    logger.info(f"{order_id} queried and canceled")


def query_and_collect(q: Query):
    if random_from_weighted(highspeed_weights):
        pairs = q.query_orders(types=tuple([1]))
    else:
        pairs = q.query_orders(types=tuple([1]), query_other=False)

    if not pairs:
        return

    # (orderId, tripId)
    pair = random_from_list(pairs)

    order_id = q.collect_order(order_id=pair[0])
    if not order_id:
        return

    logger.info(f"{order_id} queried and collected")


def query_and_execute(q: Query):
    if random_from_weighted(highspeed_weights):
        pairs = q.query_orders(types=tuple([1]))
    else:
        pairs = q.query_orders(types=tuple([1]), query_other=False)

    if not pairs:
        return

    # (orderId, tripId)
    pair = random_from_list(pairs)

    order_id = q.enter_station(order_id=pair[0])
    if not order_id:
        return

    logger.info(f"{order_id} queried and entered station")


def query_and_preserve(q: Query):
    start = ""
    end = ""
    trip_ids = []

    # v5 fix : les requêtes de billets exigent une date, sinon
    # "No Trip info content" → trip_ids=None → preserve crashe avant
    # d'atteindre ts-preserve-service. On passe une date future valide.
    date = _PRESERVE_DATE

    high_speed = random_from_weighted(highspeed_weights)
    if high_speed:
        start = "Shang Hai"
        end = "Su Zhou"
        high_speed_place_pair = (start, end)
        trip_ids = q.query_high_speed_ticket(place_pair=high_speed_place_pair, time=date)
    else:
        start = "Shang Hai"
        end = "Nan Jing"
        other_place_pair = (start, end)
        trip_ids = q.query_normal_ticket(place_pair=other_place_pair, time=date)

    _ = q.query_assurances()

    if not trip_ids:
        return  # pas de trajet ce jour-là : on n'appelle pas preserve avec None

    q.preserve(start, end, trip_ids, high_speed, date=date)


def query_and_consign(q: Query):
    if random_from_weighted(highspeed_weights):
        list = q.query_orders_all_info()
    else:
        list = q.query_orders_all_info(query_other=False)

    if not list:
        return

    # (orderId, tripId)
    res = random_from_list(list)
    order_id = q.put_consign(res)

    if not order_id:
        return

    logger.info(f"{order_id} queried and put consign")


def query_and_pay(q: Query):
    if random_from_weighted(highspeed_weights):
        pairs = q.query_orders(types=tuple([0, 1]))
    else:
        pairs = q.query_orders(types=tuple([0, 1]), query_other=False)

    if not pairs:
        return

    # (orderId, tripId)
    pair = random_from_list(pairs)
    order_id = q.pay_order(pair[0], pair[1])

    if not order_id:
        return

    logger.info(f"{order_id} queried and paid")


def query_and_rebook(q: Query):
    if random_from_weighted(highspeed_weights):
        pairs = q.query_orders(types=tuple([0, 1]))
    else:
        pairs = q.query_orders(types=tuple([0, 1]), query_other=False)

    if not pairs:
        return

    # (orderId, tripId)
    pair = random_from_list(pairs)

    order_id = q.cancel_order(order_id=pair[0])
    if not order_id:
        return

    q.rebook_ticket(pair[0], pair[1], pair[1])
    logger.info(f"{order_id} queried and rebooked")


def _ensure_contact(q: Query):
    """Garantit au moins un contact (prérequis de preserve, cf. queries.add_contact)."""
    contacts = q.query_contacts()
    if not contacts:
        q.add_contact()


def _preserve_one(q: Query):
    """Réserve un billet et retourne (order_id, trip_id) NOTPAID, ou None."""
    date = _PRESERVE_DATE
    high_speed = random_from_weighted(highspeed_weights)
    if high_speed:
        start, end = "Shang Hai", "Su Zhou"
        trip_ids = q.query_high_speed_ticket(place_pair=(start, end), time=date)
    else:
        start, end = "Shang Hai", "Nan Jing"
        trip_ids = q.query_normal_ticket(place_pair=(start, end), time=date)

    q.query_assurances()

    if not trip_ids:
        return None

    return q.preserve(start, end, trip_ids, high_speed, date=date)


def full_journey(q: Query):
    """Parcours métier complet auto-suffisant : crée un ordre puis le consomme
    dans le MÊME appel (pay→collect→enter · cancel · rebook · consign).

    Ne dépend d'AUCUN pool serveur préexistant → réveille la queue
    transactionnelle (payment, inside-payment, execute, seat, cancel, rebook,
    consign, notification, voucher) dès le premier appel, y compris sur un
    épisode dont l'état a été réinitialisé (reset_tt_state --mode deep vide les
    mongos order/preserve/payment entre épisodes).
    """
    _ensure_contact(q)

    order = _preserve_one(q)
    if not order:
        return
    order_id, trip_id = order

    branch = random_from_weighted({
        "pay_collect_enter": 60,
        "cancel": 20,
        "rebook": 10,
        "consign": 10,
    })

    if branch == "pay_collect_enter":
        if not q.pay_order(order_id, trip_id):
            return
        # voucher-service : endpoint /api/v1/voucherservice/voucher → 404 sur ce
        # build TT (vérifié au pilote). Service structurellement non-atteignable
        # → laissé imputé et documenté dans la table coverage.
        q.collect_order(order_id)
        q.enter_station(order_id)
    elif branch == "cancel":
        q.cancel_order(order_id)
    elif branch == "rebook":
        q.cancel_order(order_id)
        q.rebook_ticket(order_id, trip_id, trip_id)
    elif branch == "consign":
        infos = q.query_orders_all_info()
        if infos:
            q.put_consign(random_from_list(infos))

    logger.info(f"full_journey {order_id} done ({branch})")
