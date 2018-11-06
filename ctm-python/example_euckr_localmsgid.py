# -*- coding: cp949 -*-

"""
 vi:set et ts=4 fenc=cp949:
 Copyright (C) 2008-2010 D&SOFT
 http://open.coolsms.co.kr
"""

import sys
import coolsms


def main():
    # ��ü����
    cs = coolsms.sms()

    # ���α׷���� ������ �Է��մϴ�. (��������)
    cs.appversion("TEST/1.0")

    # �ѱ����ڵ� ����� �����մϴ�.  (������ euckr�� ����)
    # ���� ���ڵ�: euckr, utf8
    cs.charset("euckr")

    # ���̵�� �н����带 �Է��մϴ�.
    # �𿡽����������� ȸ�����Խ� �Է��� ���̵�/��й�ȣ�� �Է��մϴ�.
    cs.setuser("cs_id", "cs_passwd")

    # Local �޽���ID�� �����մϴ�.
    localkey = cs.keygen()

    # local �޽���ID �� �޽����� �����մϴ�.
    # (msgid �� �Է����� �ʴ� ��� �������� �޽���ID�� �����ؼ� �����մϴ�.)
    cs.addsms("01012341234", "0212341234", "Local Message ID �׽�Ʈ", msgid=localkey)

    if cs.connect():
        # add �� ��� �޼����� ������ �����ϴ�.
        cs.send()
    else:
        # ����ó��
        print "������ ������ �� �����ϴ�."

    # ���� ����
    cs.disconnect()

    # ��������� �����ɴϴ�.
    result = cs.getr()

    # ����� ����մϴ�.
    for i in range(len(result)):
        x = result[i]
        print "Called Number: " + x["CALLED-NUMBER"]
        print "Message ID: " + x["MESSAGE-ID"]
        print "Result Code: " + x["RESULT-CODE"]
        print "Result Message: " + x["RESULT-MESSAGE"]

    # �޸� �ʱ�ȭ
    cs.emptyall()

if __name__ == "__main__":
    main()
    sys.exit(0)
