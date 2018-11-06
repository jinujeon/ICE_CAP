# -*- coding: cp949 -*-

"""
 vi:set et ts=4 fenc=cp949:
 Copyright (C) 2008-2010 D&SOFT
 http://open.coolsms.co.kr
"""

import sys
import coolsms


def GetStatusStr(status):
    if status == "0":
        return "���۴��"
    elif status == "1":
        return "���� �� ������"
    elif status == "2":
        return "���ۿϷ�"
    else:
        return "�������� �ʴ� �޽���ID"
    
def main():
    # ��ü����
    cs = coolsms.sms()

    # ���α׷���� ������ �Է��մϴ�. (��������)
    cs.appversion("TEST/1.0")

    # �ѱ����ڵ� ����� �����մϴ�.  (������ euckr�� ����)
    # ���� ���ڵ�: euckr, utf8
    cs.charset("euckr")

    # ���̵�� �н����带 �Է��մϴ�.
    cs.setuser("cs_id", "cs_passwd")

    if cs.connect():
        # ���ۻ��¸� �о�ɴϴ�.
        # keygen() ���� ������ localkey Ȥ�� �������� ������ �޽���ID�� �Է��մϴ�.
        result = cs.rcheck("�޽���ID");
    else:
        # ����ó��
        print "������ ������ �� �����ϴ�."

    # ���� ����
    cs.disconnect()

    # ����� ����մϴ�.
    print "Status: " + GetStatusStr(result["STATUS"])
    print "Result Code: " + result["RESULT-CODE"]
    print "Result Message: " + result["RESULT-MESSAGE"]


if __name__ == "__main__":
    main()
    sys.exit(0)
