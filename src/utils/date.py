from datetime import datetime


def get_today_date_en() -> str:
    """Get today's date formatted for system message."""
    today = datetime.today()
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_of_week = day_names[today.weekday()]
    month_name_full = today.strftime("%B")
    if today.day % 10 == 1 and today.day != 11:
        day_suffix = "st"
    elif today.day % 10 == 2 and today.day != 12:
        day_suffix = "nd"
    elif today.day % 10 == 3 and today.day != 13:
        day_suffix = "rd"
    else:
        day_suffix = "th"
    return f"{day_of_week}, {month_name_full} {today.day}{day_suffix}, {today.year}"


def get_today_date_vi() -> str:
    today = datetime.today()
    day_names = [
        "Thứ hai",
        "Thứ ba",
        "Thứ tư",
        "Thứ năm",
        "Thứ sáu",
        "Thứ bảy",
        "Chủ nhật",
    ]
    day_of_week = day_names[today.weekday()]
    return f"{day_of_week}, ngày {today.day}, tháng {today.month}, năm {today.year}"