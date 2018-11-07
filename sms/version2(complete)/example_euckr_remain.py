# -*- coding: cp949 -*-

"""
 vi:set et ts=4 fenc=cp949:
 Copyright (C) 2008-2010 D&SOFT
 http://open.coolsms.co.kr
"""

import sys
import coolsms


def main():
    # ��ü�� �����մϴ�.
    cs = coolsms.sms()

    # ���α׷���� ������ �Է��մϴ�. (��������)
    cs.appversion("TEST/1.0")

    # �ѱ����ڵ� ����� �����մϴ�.  (������ euckr�� ����)
    # ���� ���ڵ�: euckr, utf8
    cs.charset("euckr")

    # ���̵�� �н����带 �Է��մϴ�.
    cs.setuser("cs_id", "cs_passwd")

    # ������ �����մϴ�.
    if cs.connect():
        # ����ũ������ ��ȸ�մϴ�.
        result = cs.remain();
    else:
        # ����ó��
        print "������ ������ �� �����ϴ�."

    # ������ �����ϴ�.
    cs.disconnect()

    # ����� ����մϴ�.
    if result["RESULT-CODE"] == "00":
        print "ĳ��:" + result["CASH"]
        print "����Ʈ:" + result["POINT"]
        print "���ڹ��:" + result["DROP"]
        print "��ü SMS �Ǽ�:" + result["CREDITS"]
    else:
        print "Result Code: " + result["RESULT-CODE"]
        print "Result Message: " + result["RESULT-MESSAGE"]


if __name__ == "__main__":
    main()
    sys.exit(0)
