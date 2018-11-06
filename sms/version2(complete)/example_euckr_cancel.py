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
    cs.setuser("cs_id", "cs_passwd")

    if cs.connect():
        # ���ۻ��¸� �о�ɴϴ�.
        # keygen() ���� ������ ���� �޽���ID(�Ǵ� �׷�ID) Ȥ��
        # �������� ������ �޽���ID(�Ǵ� �׷�ID)�� �Է��մϴ�.

        #- �޽���ID�� ���� ��ҽ�
        result = cs.cancel("20101021133888234878558637");

        #- �׷�ID�� ���� ��ҽ�
        result = cs.groupcancel("20101021133888234878558637");
    else:
        # ����ó��
        print "������ ������ �� �����ϴ�."

    # ���� ����
    cs.disconnect()

    # ����� ����մϴ�.
    print "Result Code: " + result["RESULT-CODE"]
    print "Result Message: " + result["RESULT-MESSAGE"]

    # �޸� �ʱ�ȭ
    cs.emptyall()


if __name__ == "__main__":
    main()
    sys.exit(0)
