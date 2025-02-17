def create_precaution_comment(category_name:str):
    comment = ''
    if category_name in ["fashion", "goods"]:
        comment = "패션/의류 제품군은 트렌드 변화에 민감하여 재고 폐기 비용 증가에 주의해야 합니다."
    elif category_name in ["digital"]:
        comment = "디지털/가전 제품군은 신제품 출시 주기가 짧고, 기술 변화가 빠르므로 재고 과잉으로 인한 감가상각 손실에 주의해야 합니다."
    elif category_name in ["food"]:
        comment = "식품 제품군은 유통기한이 짧고, 수요 변동성이 크므로 충분한 재고 수준을 유지하는 동시에 폐기율을 최소화할 수 있도록 관리해야 합니다."
    elif category_name in ["cosmetic", "life", "other", "book", "sports", "baby"]:
        comment = "해당 품목군은 비교적 안정적인 수요를 가지지만, 대량 주문 발생 가능성이 있으므로 리드타임을 고려한 재고 확보가 필요합니다."
    elif category_name in ["furniture"]:
        comment = "가구/인테리어 제품군은 부피가 크고 회전율이 낮으므로 보관 비용 증가에 주의해야 합니다."
    
    return comment